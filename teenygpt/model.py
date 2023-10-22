import math
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm  # type: ignore

from .config import (
    ModelConfig,
    TrainConfig,
    PositionalEncodingType,
    FFNType,
    LayerNormType,
)
from .dataset import Dataset


class Model(nn.Module):
    """
    Base model class that contains some helper functions shared by all models.

    Attributes:
        config: The configuration of the model.
        name: The name of the model (for debugging purposes only).
    """

    config: ModelConfig
    name: str

    def __init__(self, config: ModelConfig, name: str) -> None:
        super().__init__()
        self.name = name
        self.config = config

    def param_count(self) -> int:
        """
        Returns the total number of parameter elements used by the model.
        """
        return sum(m.numel() for m in self.parameters())

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Computes the cross-entropy loss between the predicted unnormalized logits and
        the ground truth targets.

        Args:
            logits: The unnormalized logits tensor, as produced by the forward method.
                This tensor is expected to have shape ``(n_batch, n_context, n_vocab)``
                as defined by the config.
            targets: The target token indices. This tensor is expected to have shape
                ``(n_batch, n_context)`` as defined by the config.

        Returns:
            A scalar tensor containing the loss.
        """
        return F.cross_entropy(
            logits.reshape(-1, self.config.n_vocab), targets.flatten()
        )

    def estimate_loss(self, dataset: Dataset, config: TrainConfig) -> float:
        """
        Estimates loss with a given dataset and number batches.

        Args:
            dataset: The Dataset that should be used to estimate the loss.
            config: The training config specifying how loss should be estimated.

        Returns:
            The estimated loss.
        """
        with self._eval_mode():
            loss = 0.0
            for _ in range(config.n_est_loss_batches):
                xs, ys = dataset.get_batch(config.n_batch, config.n_context)
                logits = self.forward(xs)
                loss += self.loss(logits, ys).item()
            return loss / config.n_est_loss_batches

    def generate(self, inputs: torch.Tensor, output_len: int) -> torch.Tensor:
        """
        Generates continuations from the given inputs. The sampling is simple greedy
        sampling (with zero temperature).

        Args:
            inputs: The input sequences tensor. This tensor is expected to have shape
                ``(n_seq, input_len)`` where ``n_seq`` is the number of input sequences
                 and ``input_len`` is the length of each input. This tensor should
                contain token indices.
            output_len: The length of the continuation that should be generated
                for each input sequence.

        Args:
            An output tensor containing token indices for the continuation, with shape
            ``(n_seq, output_len)``.
        """
        with self._eval_mode():
            generated = inputs
            input_len = inputs.size(-1)
            for _ in tqdm(range(output_len)):
                window = generated[:, -input_len:]
                logits = self.forward(window)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, samples], dim=-1)
            return generated[:, input_len:]

    @contextmanager
    def _eval_mode(self):
        """
        Context manager which sets the model to evaluation mode and disables
        gradient calculations.
        """
        was_training = self.training
        self.train(False)
        try:
            with torch.inference_mode():
                yield
        finally:
            self.train(was_training)


class NaiveModel(Model):
    """
    A baseline model which contains a single linear layer with a ReLU activation.
    """

    def __init__(self, config: ModelConfig, name: str = "NaiveModel") -> None:
        super().__init__(config, name)
        self.embedding = nn.Embedding(config.n_vocab, config.n_dim)
        self.linear = nn.Sequential(
            nn.Linear(config.n_dim, config.n_dim),
            nn.ReLU(),
            nn.Linear(config.n_dim, config.n_vocab),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        return x


class ClassicFFNBlock(nn.Module):
    """
    "Classic" attention feed-forward network.

    This implements the attention FFN block from the original Transformer paper.
    """

    def __init__(self, n_dim: int, p_dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim, n_dim),
            nn.Dropout(p_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SwishGatedLinearUnitFFNBlock(nn.Module):
    """
    Swish GLU attention feed-forward network.

    This block utilizes the FFN_SwiGLU function defined in "GLU Variants Improve
    Transformer" (https://arxiv.org/pdf/2002.05202v1.pdf).
    """

    def __init__(self, n_dim: int, p_dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(n_dim, n_dim, bias=False)
        self.w2 = nn.Linear(n_dim, n_dim, bias=False)
        self.w3 = nn.Linear(n_dim, n_dim, bias=False)
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSLayerNorm(torch.nn.Module):
    """
    RMS layer normalization.

    Implementation of "Root Mean Square Layer Normalization" (https://arxiv.org/abs/1910.07467).
    """

    def __init__(self, n_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))

    def forward(self, x):
        # Compute mean squared.
        ms = (x**2).mean(-1, keepdim=True)

        # Compute inverse root mean squared.
        # N.B., need to add epsilon in case ms is zero for numerical stability
        irms = torch.rsqrt(ms + 1e-8)

        return x * self.weight * irms


class AttentionBlock(nn.Module):
    """
    The multi-head attention block.

    We use a pre-layer-norm (as opposed to post-layer-norm) architecture, as current
    research suggests that pre-LN architectures are easier to train (e.g., see
    https://arxiv.org/abs/2304.14802). Additionally, in practice, both Llama 2 and GPT
    models seem to use pre-norm architectures.

    This function uses the beta PyTorch ``scaled_dot_product_attention`` API for
    computing attention efficiently via optimized kernels.
    """

    _attention_mask: torch.Tensor
    _rope_matrix: torch.Tensor | None
    ffn: nn.Module

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Compute the dimension of each attention head.
        self.n_head_dim = self.config.n_dim // self.config.n_attn_heads
        if self.n_head_dim * self.config.n_attn_heads != self.config.n_dim:
            raise ValueError(
                "dimension of input is not cleanly divisible by number of heads"
            )

        # Linear weight matrices for attention queries, keys, values, and output.
        self.linear_q = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.linear_k = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.linear_v = nn.Linear(config.n_dim, config.n_dim, bias=False)
        self.linear_o = nn.Linear(config.n_dim, config.n_dim, bias=False)

        # Layer norms.
        self.norm_1 = self._init_layer_norm(config)
        self.norm_2 = self._init_layer_norm(config)

        # Feed-forward network.
        match config.ffn_type:
            case FFNType.CLASSIC:
                self.ffn = ClassicFFNBlock(config.n_dim, config.p_dropout)
            case FFNType.SWISH_GLU:
                self.ffn = SwishGatedLinearUnitFFNBlock(config.n_dim, config.p_dropout)
            case _:
                raise ValueError(f"unexpected ffn_type {config.ffn_type}")

        # Generate attention mask.
        if config.positional_encoding_type == PositionalEncodingType.ALIBI:
            self._attention_mask = _generate_alibi_mask(
                config.n_context_max, config.n_attn_heads
            )
        else:
            self._attention_mask = _generate_default_mask(config.n_context_max)

        # Generate RoPE matrix.
        if config.positional_encoding_type == PositionalEncodingType.ROPE:
            self._rope_matrix = _generate_rope_matrix(
                config.n_context_max, self.n_head_dim
            )
        else:
            self._rope_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention residual block.
        x = x + self._multihead_attention(self.norm_1(x))

        # Feed-forward network residual block.
        x = x + self.ffn(self.norm_2(x))

        return x

    def _multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of multi-head attention from "Attention is All You Need"
        (https://arxiv.org/abs/1706.03762).
        """
        n_batch, n_context, n_dim = x.shape
        if n_dim != self.config.n_dim:
            raise ValueError("input tensor has wrong embedding dimension")

        # Perform linear projections to get the queries, keys, and values.
        # N.B., the following matrix multiplies could be batched into a single multiply.
        q_proj = self.linear_q(x)
        k_proj = self.linear_k(x)
        v_proj = self.linear_v(x)

        # Rearrange the query, key, and value tensors to have shape
        # (n_batch, n_attn_heads, n_context, n_head_dim) so that we can apply
        # scaled_dot_product_attention.
        q_proj_heads = q_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)
        k_proj_heads = k_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)
        v_proj_heads = v_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)

        # Apply RoPE matrix if enabled.
        if self._rope_matrix is not None:
            q_proj_heads = q_proj_heads.unsqueeze(-1)
            q_proj_heads = self._rope_matrix @ q_proj_heads
            q_proj_heads = q_proj_heads.squeeze(-1)

            k_proj_heads = k_proj_heads.unsqueeze(-1)
            k_proj_heads = self._rope_matrix @ k_proj_heads
            k_proj_heads = k_proj_heads.squeeze(-1)

        # Apply our attention function.
        o_heads = F.scaled_dot_product_attention(
            q_proj_heads,
            k_proj_heads,
            v_proj_heads,
            attn_mask=self._attention_mask[:, :n_context, :n_context],
            dropout_p=self.config.p_dropout,
        )

        # Rearrange the output to conform to the original shape.
        o = o_heads.permute(0, 2, 1, 3).reshape(n_batch, n_context, n_dim)

        # Perform a final linear projection on the output.
        return self.linear_o(o)

    def _init_layer_norm(self, config: ModelConfig) -> nn.Module:
        match config.layer_norm_type:
            case LayerNormType.NONE:
                return nn.Identity()
            case LayerNormType.CLASSIC:
                return nn.LayerNorm(config.n_dim)
            case LayerNormType.RMS:
                return RMSLayerNorm(config.n_dim)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as per the original Transformer
    paper (https://arxiv.org/abs/1706.03762).
    """

    def __init__(self, n_context_max: int, n_dim: int) -> None:
        super().__init__()
        self.encoding = _generate_sinusoidal_encoding(n_context_max, n_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, n_context, n_dim = x.shape
        return x + self.encoding[:n_context, :].unsqueeze(0)


class LearnedEmbeddingPositionalEncoding(nn.Module):
    """
    Positional encoding via a learned embedding.
    """

    def __init__(self, n_context_max: int, n_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_context_max, n_dim)
        self.positions = torch.arange(n_context_max, dtype=torch.int64).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, n_context, n_dim = x.shape
        return x + self.embedding(self.positions[:, :n_context])


class LearnedSinusoidalPositionalEncoding(nn.Module):
    """
    Learned sinusoidal positional encoding.
    """

    def __init__(
        self, n_context_max: int, n_dim: int, n_dim_sinusoid: int = 16
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_dim_sinusoid, n_dim)
        self.positions = _generate_sinusoidal_encoding(n_context_max, n_dim_sinusoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_batch, n_context, n_dim = x.shape
        return x + self.linear(self.positions[:n_context, :]).unsqueeze(0)


class TransformerModel(Model):
    """
    A transformer model.
    """

    positional_encoding: nn.Module

    def __init__(self, config: ModelConfig, name: str = "TransformerModel") -> None:
        super().__init__(config, name)
        self.embedding = nn.Embedding(config.n_vocab, config.n_dim)
        self.unembedding = nn.Linear(config.n_dim, config.n_vocab)

        match config.positional_encoding_type:
            case PositionalEncodingType.SINUSOIDAL:
                self.positional_encoding = SinusoidalPositionalEncoding(
                    config.n_context_max, config.n_dim
                )
            case PositionalEncodingType.LEARNED_EMBEDDING:
                self.positional_encoding = LearnedEmbeddingPositionalEncoding(
                    config.n_context_max, config.n_dim
                )
            case PositionalEncodingType.LEARNED_SINUSOIDAL:
                self.positional_encoding = LearnedSinusoidalPositionalEncoding(
                    config.n_context_max, config.n_dim
                )
            case _:
                self.positional_encoding = nn.Identity()

        self.attention_blocks = nn.Sequential(
            OrderedDict(
                [
                    (f"attention_{i}", AttentionBlock(config))
                    for i in range(config.n_attn_layers)
                ]
            )
        )

        self.dropout = nn.Dropout(config.p_dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        x = self.attention_blocks(x)
        x = self.unembedding(x)
        return x


def _generate_alibi_mask(n_context_max: int, n_heads: int) -> torch.Tensor:
    """
    Generates an ALiBi attention mask.

    Args:
        n_context_max: The maximum context window length.
        n_heads: The number of attention heads to generate masks for.

    Returns:
        An attention mask tensor with shape ``(n_heads, n_context_max, n_context_max)``.
    """
    ratio = 2 ** (-8 / n_heads)
    base_mask = _generate_alibi_mask_single_head(n_context_max)
    return torch.stack([base_mask * (ratio ** (i + 1)) for i in range(n_heads)])


def _generate_alibi_mask_single_head(n_context_max: int) -> torch.Tensor:
    def _el(i, j):
        if i > j:
            return -math.inf
        else:
            return -abs(i - j)

    return torch.tensor(
        [[_el(i, j) for i in range(n_context_max)] for j in range(n_context_max)]
    )


def _generate_default_mask(n_context_max: int) -> torch.Tensor:
    """
    Generates a default attention mask.

    This should generate the same mask as
    ``torch.nn.Transformer.generate_square_subsequent_mask``. However, the
    implementation of the above function in PyTorch 2.10 currently seems to have a bug
    which it to not respect the default device and dtype correctly. Therefore,
    we have our own implementation.

    Args:
        n_context_max: The maximum context window length.

    Returns:
        An attention mask tensor with shape ``(1, n_context_max, n_context_max)``.
    """
    m = torch.full((n_context_max, n_context_max), -torch.inf)
    m = torch.triu(m, diagonal=1)
    m = m.unsqueeze(0)
    return m


def _generate_sinusoidal_encoding(n_context_max: int, n_dim: int) -> torch.Tensor:
    """
    Generates a sinusoidal positional encoding as per the original Transformer
    paper (https://arxiv.org/abs/1706.03762).

    Args:
        n_context_max: The maximum context window length.
        n_dim: The model dimension that the encoding should be generated for.

    Returns:
        A tensor with shape ``(n_context_max, n_dim)`` containing the encoding.
    """

    # Verify dimension is evenly divisible by 2.
    n_dim_half = n_dim // 2
    if n_dim != n_dim_half * 2:
        raise ValueError("expected dimension to be divisible by two")

    # Generate points to evaluate sin/cos, with shape (n_context_max, n_dim_half).
    periods = 10000 ** (np.arange(n_dim_half) * 2 / n_dim)
    pos = np.arange(n_context_max)
    p = np.outer(pos, 1 / periods)

    # Evaluate sin/cos at each point.
    p_sin = np.sin(p)
    p_cos = np.cos(p)

    # Interleave sin/cos, resulting in shape (n_context_max, n_dim).
    p_interleaved = (
        np.concatenate([p_sin.T, p_cos.T], axis=-1).reshape(n_dim, n_context_max).T
    )
    return torch.tensor(p_interleaved, dtype=torch.get_default_dtype())


def _generate_rope_matrix(n_context_max: int, n_head_dim: int) -> torch.Tensor:
    """
    Generates a RoPE rotary matrix as per the RoFormer paper (https://arxiv.org/abs/2104.09864).

    Args:
        n_context_max: The maximum context window length.
        n_head_dim: The attention head dimension.

    Returns:
        A tensor with shape ``(n_context_max, n_head_dim, n_head_dim)``.
    """
    R = torch.zeros((n_context_max, n_head_dim, n_head_dim))
    for m in range(n_context_max):
        for i in range(n_head_dim // 2):
            theta_i = 10000.0 ** (-2.0 * (i - 1) / n_head_dim)
            cos = np.cos(m * theta_i)
            sin = np.sin(m * theta_i)
            R[m, 2 * i, 2 * i] = cos
            R[m, 2 * i, 2 * i + 1] = -sin
            R[m, 2 * i + 1, 2 * i] = sin
            R[m, 2 * i + 1, 2 * i + 1] = cos
    return R
