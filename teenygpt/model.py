import math
from collections import OrderedDict
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .dataset import Dataset


class Model(nn.Module):
    config: Config

    def __init__(self, config: Config, name: str) -> None:
        super().__init__()
        self.name = name
        self.config = config

    def param_count(self) -> int:
        return sum(m.numel() for m in self.parameters())

    def loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.flatten()
        )

    def estimate_loss(self, dataset: Dataset, batch_count: int = 10) -> float:
        with self._eval_mode():
            loss = 0.0
            for _ in range(batch_count):
                xs, ys = dataset.get_batches(self.config)
                logits = self.forward(xs)
                loss += self.loss(logits, ys).item()
            return loss / batch_count

    def generate(self, inputs: torch.Tensor, count: int = 30) -> torch.Tensor:
        with self._eval_mode():
            generated = inputs
            for _ in range(count):
                window = generated[:, -self.config.context_window :]
                logits = self.forward(window)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                samples = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, samples], dim=-1)
            return generated

    @contextmanager
    def _eval_mode(self):
        was_training = self.training
        self.train(False)
        try:
            with torch.inference_mode():
                yield
        finally:
            self.train(was_training)


class NaiveModel(Model):
    def __init__(self, config: Config) -> None:
        super().__init__(config, "NaiveModel")
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.linear = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        a = self.embedding(input)
        a = self.layer_norm(a)
        a = self.linear(a)
        a = a @ self.embedding.weight.T
        return a


class RMSNorm(nn.Module):
    def __init__(self, layer_shape):
        super().__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(layer_shape)))

    def forward(self, x):
        """
        assumes shape is (batch, seq_len, d_model)
        """
        # frob norm is not the same as RMS. RMS = 1/sqrt(N) * frob norm
        ff_rms = torch.linalg.norm(x, dim=(1, 2)) * x[0].numel() ** -0.5
        raw = x / ff_rms.unsqueeze(-1).unsqueeze(-1)
        return self.scale[: x.shape[1], :].unsqueeze(0) * raw


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit
    https://arxiv.org/pdf/2002.05202v1.pdf
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.linear_gate = nn.Linear(size, size)
        self.linear = nn.Linear(size, size)
        self.beta = torch.randn(1, requires_grad=True)

        self.beta = nn.Parameter(torch.ones(1))
        self.register_parameter("beta", self.beta)

    def forward(self, x):
        swish_gate = self.linear_gate(x) * torch.sigmoid(
            self.beta * self.linear_gate(x)
        )
        out = swish_gate * self.linear(x)
        return out


class AttentionBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.linear_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.linear_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.linear_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.linear_o = nn.Linear(config.d_model, config.d_model, bias=False)

        # self.linear_q_2 = nn.Linear(config.d_model, config.d_model)
        # self.linear_k_2 = nn.Linear(config.d_model, config.d_model)
        # self.linear_v_2 = nn.Linear(config.d_model, config.d_model)
        # self.linear_o_2 = nn.Linear(config.d_model, config.d_model)

        # self.multihead_attention = nn.MultiheadAttention(
        #     config.d_model,
        #     config.attention_heads,
        #     dropout=config.dropout_p,
        #     batch_first=True,
        # )

        # self.layer_norm_1 = nn.LayerNorm(config.d_model)
        # self.layer_norm_2 = nn.LayerNorm(config.d_model)

        self.rms_norm_1 = RMSNorm((config.context_window, config.d_model))
        self.rms_norm_2 = RMSNorm((config.context_window, config.d_model))

        self.linear = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SwiGLU(config.d_model),
        )

        self.rope_matrix = _generate_rope_matrix(
            config.context_window, config.d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention w/ pre-layer-norm.
        x_attn = self._multihead_attention(self.rms_norm_1(x))

        # Residual block #1.
        x_res_1 = x + x_attn

        # Feed-forward w/ pre-layer-norm.
        x_ffn = self.linear(self.rms_norm_2(x_res_1))

        # Residual block #2.
        x_res_2 = x_res_1 + x_ffn

        return x_res_2

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        return (
            torch.bmm(x.transpose(0, 1), self.rope_matrix[: x.size(1), ...])
        ).transpose(0, 1)

    # def _multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
    #     _, window_size, _ = x.shape

    #     activations, _ = self.multihead_attention(
    #         # N.B., the following hack disables the fast path, which is incompatible
    #         # with a non-boolean attention mask.
    #         x.view(x.shape),
    #         x,
    #         x,
    #         attn_mask=_generate_alibi_mask_single_head(window_size),
    #         is_causal=True,
    #     )

    #     return activations

    def _multihead_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window_size, _ = x.shape

        # N.B., the following matrix multiplies could be batched.
        q_proj = self.linear_q(x)
        k_proj = self.linear_k(x)
        v_proj = self.linear_v(x)

        q_proj = self._apply_rope(q_proj)
        k_proj = self._apply_rope(k_proj)
        v_proj = self._apply_rope(v_proj)

        # q_proj = self.linear_q_2(q_proj)
        # k_proj = self.linear_k_2(k_proj)
        # v_proj = self.linear_v_2(v_proj)

        head_dim = self.config.d_model // self.config.attention_heads
        if head_dim * self.config.attention_heads != self.config.d_model:
            raise ValueError(
                "dimension of input is not cleanly divisible by number of heads"
            )

        attn_mask = _generate_alibi_mask(
            window_size, self.config.attention_heads
        )
        # attn_mask = _generate_alibi_mask_single_head(window_size)

        q_proj_heads = q_proj.view(
            batch_size, window_size, self.config.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        k_proj_heads = k_proj.view(
            batch_size, window_size, self.config.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        v_proj_heads = v_proj.view(
            batch_size, window_size, self.config.attention_heads, head_dim
        ).permute(0, 2, 1, 3)
        o_heads = F.scaled_dot_product_attention(
            q_proj_heads,
            k_proj_heads,
            v_proj_heads,
            attn_mask=attn_mask,
            dropout_p=self.config.dropout_p,
        )
        o = o_heads.permute(0, 2, 1, 3).reshape(
            batch_size, window_size, self.config.d_model
        )

        return self.linear_o(o)


class AttentionModel(Model):
    def __init__(self, config: Config, name: str = "AttentionModel") -> None:
        super().__init__(config, name)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.attention_blocks = nn.Sequential(
            OrderedDict(
                [
                    (f"attention_{i}", AttentionBlock(config))
                    for i in range(config.attention_layers)
                ]
            )
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SwiGLU(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embedding(inputs)
        x = self.attention_blocks(x)
        return self.ffn(x)


def _generate_alibi_mask(n, n_heads):
    ratio = 2 ** (-8 / n_heads)
    base_mask = _generate_alibi_mask_single_head(n)
    return torch.stack(
        [base_mask * (ratio ** (i + 1)) for i in range(n_heads)]
    )


def _generate_alibi_mask_single_head(n):
    def _el(i, j):
        if i > j:
            return -math.inf
        else:
            return -abs(i - j)

    return torch.tensor([[_el(i, j) for i in range(n)] for j in range(n)])


def _generate_rope_matrix(context_window, embedding_dim):
    R = torch.zeros(
        (context_window, embedding_dim, embedding_dim), requires_grad=False
    )
    for position in range(context_window):
        for i in range(embedding_dim // 2):
            theta = 10000.0 ** (-2.0 * (i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2 * i, 2 * i] = math.cos(m_theta)
            R[position, 2 * i, 2 * i + 1] = -math.sin(m_theta)
            R[position, 2 * i + 1, 2 * i] = math.sin(m_theta)
            R[position, 2 * i + 1, 2 * i + 1] = math.cos(m_theta)
    return R
