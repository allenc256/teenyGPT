import math
import torch
from torch import nn
from torch.nn import functional as F


def _generate_alibi_mask(n):
    def _el(i, j):
        if i > j:
            return -math.inf
        else:
            return -abs(i - j)

    return torch.tensor([[_el(i, j) for i in range(n)] for j in range(n)])


q = torch.arange(5 * 3 * 4, dtype=torch.float).reshape(5, 3, 4)
k = torch.arange(5 * 3 * 4, dtype=torch.float).reshape(5, 3, 4)
v = torch.arange(5 * 3 * 4, dtype=torch.float).reshape(5, 3, 4)

attention = nn.MultiheadAttention(
    embed_dim=q.size(-1),
    num_heads=1,
    batch_first=False,
    bias=False,
    dropout=0.0,
)

torch.set_printoptions(precision=4, sci_mode=False)

w_q = attention.in_proj_weight[:4].T
w_k = attention.in_proj_weight[4:8].T
w_v = attention.in_proj_weight[8:].T
w_o = attention.out_proj.weight.T

q_proj = q @ w_q
k_proj = k @ w_k
v_proj = v @ w_v

# attn_mask = nn.Transformer.generate_square_subsequent_mask(3)
attn_mask = _generate_alibi_mask(3)
# attn_mask = torch.zeros(3, 3)

with torch.no_grad():
    attention.eval()

    o = F.softmax(
        (
            (q_proj @ k_proj.transpose(-2, -1)) / math.sqrt(q.size(-1))
            + attn_mask
        ),
        dim=-1,
    )
    o = o @ v_proj
    o = o @ w_o

    o_torch1 = F.scaled_dot_product_attention(
        q_proj, k_proj, v_proj, attn_mask=attn_mask
    )
    o_torch1 = o_torch1 @ w_o

    q_b2 = q.permute(1, 0, 2)
    print(q.shape, q_b2.shape)
    o_torch2, _ = attention(
        q_b2, q_b2, q_b2, attn_mask=attn_mask, need_weights=False
    )
    o_torch2 = o_torch2.permute(1, 0, 2)

    print(attn_mask)
    print(o[0])
    print(o_torch1[0])
    print(o_torch2[0])
    print(o.shape, o_torch1.shape, o_torch2.shape)
