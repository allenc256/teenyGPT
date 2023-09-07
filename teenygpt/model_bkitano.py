import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from .config import Config
from .model import Model


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


class ALiBiAttention(nn.Module):
    config: Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.multihead = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=0.1,
            batch_first=True,
        )

    def forward(self, x, return_attn_weights=False):
        _, m, _ = x.shape

        activations, attn_weights = self.multihead(
            x.view(x.shape),
            x,
            x,
            attn_mask=_generate_alibi_mask_single_head(m),
        )

        if return_attn_weights:
            return activations, attn_weights
        return activations


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


class RoPEAttention_wMask(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=False)

        self.multihead = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.R = self.get_rotary_matrix(config.context_window, config.d_model)

    def get_rotary_matrix(self, context_window, embedding_dim):
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

    def forward(self, x, return_attn_weights=False):
        b, m, d = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q_out = (torch.bmm(q.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)
        k_out = (torch.bmm(k.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)
        v_out = (torch.bmm(v.transpose(0, 1), self.R[:m, ...])).transpose(0, 1)

        activations, attn_weights = self.multihead(
            q_out,
            k_out,
            v_out,
            attn_mask=nn.Transformer.generate_square_subsequent_mask(m),
            is_causal=True,
        )

        if return_attn_weights:
            return activations, attn_weights
        return activations


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


class RopeModel(Model):
    def __init__(self, config: Config) -> None:
        super().__init__(config, "RopeModel")

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.rms = RMSNorm((config.context_window, config.d_model))
        self.rope_attention = RoPEAttention_wMask(config)

        self.linear = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SwiGLU(config.d_model),
        )

        self.last_linear = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, idx, targets=None):
        x = self.embedding(idx)

        # one block of attention
        x = self.rms(x)  # rms pre-normalization
        x = x + self.rope_attention(x)

        x = self.rms(x)  # rms pre-normalization
        x = x + self.linear(x)

        logits = self.last_linear(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )
            return logits, loss

        else:
            return logits


# add RMSNorm and residual conncection
class LlamaBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

        self.rms = RMSNorm((config.context_window, config.d_model))

        self.attention = RoPEAttention_wMask(config)
        # self.attention = ALiBiAttention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SwiGLU(config.d_model),
        )

    def forward(self, x):
        x = self.rms(x)  # rms pre-normalization
        x = x + self.attention(x)

        x = self.rms(x)  # rms pre-normalization
        x = x + self.feedforward(x)
        return x


class LlamaModel(Model):
    def __init__(self, config: Config, name: str = "LlamaModel") -> None:
        super().__init__(config, name)
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.llama_blocks = nn.Sequential(
            OrderedDict(
                [
                    (f"llama_{i}", LlamaBlock(config))
                    for i in range(config.attention_layers)
                ]
            )
        )

        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            SwiGLU(config.d_model),
            nn.Linear(config.d_model, config.vocab_size),
        )

    def forward(self, idx, targets=None):
        x = self.embeddings(idx)
        x = self.llama_blocks(x)
        logits = self.ffn(x)

        if targets is None:
            return logits

        else:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size), targets.view(-1)
            )
            return logits, loss
