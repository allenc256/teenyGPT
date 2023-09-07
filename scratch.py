import math
import torch
from torch import nn
from torch.nn import functional as F

q = torch.arange(3 * 4, dtype=torch.float).reshape(3, 4)
k = torch.arange(3 * 4, dtype=torch.float).reshape(3, 4)
v = torch.arange(3 * 4, dtype=torch.float).reshape(3, 4)

attention = nn.MultiheadAttention(
    embed_dim=q.size(-1), num_heads=2, batch_first=True, bias=False
)

torch.set_printoptions(precision=4, sci_mode=False)

w_q = attention.in_proj_weight[:4].T
w_k = attention.in_proj_weight[4:8].T
w_v = attention.in_proj_weight[8:].T
w_o = attention.out_proj.weight.T

q_proj = q @ w_q
k_proj = k @ w_k
v_proj = v @ w_v

q_proj_heads = q_proj.view(3, 2, 2).permute(1, 0, 2)
k_proj_heads = k_proj.view(3, 2, 2).permute(1, 0, 2)
v_proj_heads = v_proj.view(3, 2, 2).permute(1, 0, 2)
o_heads = F.scaled_dot_product_attention(
    q_proj_heads, k_proj_heads, v_proj_heads
)
o = o_heads.permute(1, 0, 2).reshape(3, 4) @ w_o

o_torch = attention(q, k, v, need_weights=False)

print(o_torch)
print(o)
