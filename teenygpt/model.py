import math
from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm  # type: ignore

from .config import ModelConfig, TrainConfig, PositionalEncoding
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


class FeedForwardBlock(nn.Module):
    """
    The attention feed-forward block.

    This block utilizes the FFN_SwiGLU function defined in "GLU Variants Improve
    Transformer" (https://arxiv.org/pdf/2002.05202v1.pdf). This function was chosen
    because it was used in Llama 2 and appears to provide some improvement over
    simpler FFN schemes.
    """

    def __init__(self, n_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(n_dim, n_dim, bias=False)
        self.w2 = nn.Linear(n_dim, n_dim, bias=False)
        self.w3 = nn.Linear(n_dim, n_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AttentionBlock(nn.Module):
    """
    The multi-head attention block.

    This multi-head attention implementation uses ALiBi for its positional encoding (see
    https://arxiv.org/abs/2108.12409 for more details). ALiBi was chosen over potential
    alternatives for its purported ability to better generalize across arbitrary context
    lengths.

    We use a pre-layer-norm (as opposed to post-layer-norm) architecture, as current
    research suggests that pre-LN architectures are easier to train (e.g., see
    https://arxiv.org/abs/2304.14802). Additionally, in practice, both Llama 2 and GPT
    models seem to use pre-norm architectures.

    This function uses the beta PyTorch ``scaled_dot_product_attention`` API for
    computing attention efficiently via optimized kernels.
    """

    _attention_mask: torch.Tensor

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
        # N.B., it might be better to do layer norm w/out bias here.
        self.norm_1 = nn.LayerNorm(config.n_dim)
        self.norm_2 = nn.LayerNorm(config.n_dim)

        # Feed-forward network.
        self.ffn = FeedForwardBlock(config.n_dim)

        # Generate attention mask.
        self._attention_mask = self._generate_attention_mask(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention residual block.
        x = x + self._multihead_attention(self.norm_1(x))

        # Feed-forward network residual block.
        x = x + self.ffn(self.norm_2(x))

        return x

    def _generate_attention_mask(self, config: ModelConfig) -> torch.Tensor:
        match self.config.positional_encoding:
            case PositionalEncoding.ALIBI:
                return _generate_alibi_mask(config.n_context_max, config.n_attn_heads)
            case (
                PositionalEncoding.NONE
                | PositionalEncoding.SINUSOIDAL
                | PositionalEncoding.LEARNED
            ):
                return _generate_default_mask(config.n_context_max)
            case _:
                raise ValueError(
                    f"unsupported encoding: {self.config.positional_encoding}"
                )

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
        # (..., n_window, n_head_dim) so that we can apply scaled_dot_product_attention.
        q_proj_heads = q_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)
        k_proj_heads = k_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)
        v_proj_heads = v_proj.view(
            n_batch, n_context, self.config.n_attn_heads, self.n_head_dim
        ).permute(0, 2, 1, 3)

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


class LearnedPositionalEncoding(nn.Module):
    """
    Positional encoding via a learned embedding.
    """

    def __init__(self, n_context_max: int, n_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_context_max, n_dim)
        self.positions = torch.arange(n_context_max, dtype=torch.int64).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.embedding(self.positions[:, : x.size(1)])


class TransformerModel(Model):
    """
    A transformer model.
    """

    positional_encoding: nn.Module

    def __init__(self, config: ModelConfig, name: str = "TransformerModel") -> None:
        super().__init__(config, name)
        self.embedding = nn.Embedding(config.n_vocab, config.n_dim)
        self.unembedding = nn.Linear(config.n_dim, config.n_vocab)

        match config.positional_encoding:
            case PositionalEncoding.SINUSOIDAL:
                self.positional_encoding = SinusoidalPositionalEncoding(
                    config.n_context_max, config.n_dim
                )
            case PositionalEncoding.LEARNED:
                self.positional_encoding = LearnedPositionalEncoding(
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(inputs)
        x = self.positional_encoding(x)
        x = self.attention_blocks(x)
        x = self.unembedding(x)
        return x


def _generate_alibi_mask(n_context: int, n_heads: int) -> torch.Tensor:
    """
    Generates an ALiBi attention mask.

    Args:
        n_context: The size of the context window to generate a mask for.
        n_heads: The number of attention heads to generate masks for.

    Returns:
        An attention mask tensor with shape ``(n_heads, n_context, n_context)``.
    """
    ratio = 2 ** (-8 / n_heads)
    base_mask = _generate_alibi_mask_single_head(n_context)
    return torch.stack([base_mask * (ratio ** (i + 1)) for i in range(n_heads)])


def _generate_alibi_mask_single_head(n_context: int) -> torch.Tensor:
    def _el(i, j):
        if i > j:
            return -math.inf
        else:
            return -abs(i - j)

    return torch.tensor(
        [[_el(i, j) for i in range(n_context)] for j in range(n_context)]
    )


def _generate_default_mask(n_context: int) -> torch.Tensor:
    """
    Generates a default attention mask.

    This should generate the same mask as
    ``torch.nn.Transformer.generate_square_subsequent_mask``. However, the
    implementation of the above function in PyTorch 2.10 currently seems to have a bug
    which it to not respect the default device and dtype correctly. Therefore,
    we have our own implementation.

    Args:
        n_context: The size of the context window to generate a mask for.

    Returns:
        An attention mask tensor with shape ``(1, n_context, n_context)``.
    """
    m = torch.full((n_context, n_context), -torch.inf)
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
    p = pos.reshape(-1, 1) / periods.reshape((1, -1))

    # Evaluate sin/cos at each point.
    p_sin = np.sin(p)
    p_cos = np.cos(p)

    # Interleave sin/cos, resulting in shape (n_context_max, n_dim).
    p_interleaved = (
        np.concatenate([p_sin.T, p_cos.T], axis=-1).reshape(n_dim, n_context_max).T
    )
    return torch.tensor(p_interleaved, dtype=torch.get_default_dtype())
