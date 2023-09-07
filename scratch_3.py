import math
import torch
from torch import nn
from torch.nn import functional as F


def _generate_alibi_mask(n_dim, n_heads):
    ratio = 2 ** (-8 / n_heads)
    base_mask = _generate_alibi_mask_single_head(n_dim)
    return torch.stack(
        [base_mask * (ratio ** (i + 1)) for i in range(n_heads)]
    )


def _generate_alibi_mask_single_head(n_dim):
    def _el(i, j):
        if i > j:
            return -math.inf
        else:
            return -abs(i - j)

    return torch.tensor(
        [[_el(i, j) for i in range(n_dim)] for j in range(n_dim)]
    )


print(_generate_alibi_mask(3, 4))

# x = torch.arange(2 * 2 * 3 * 2, dtype=torch.float).reshape(2, 2, 3, 2) / 100

# torch.set_printoptions(precision=4, sci_mode=False)

# attn_mask = torch.stack([_generate_alibi_mask(3), torch.eye(3, 3)])

# o = F.scaled_dot_product_attention(x, x, x, attn_mask=attn_mask)
# print(o[1, 0])
# print(o[1, 1])

# o = F.scaled_dot_product_attention(
#     x[1, 0], x[1, 0], x[1, 0], attn_mask=attn_mask[0]
# )
# print(o)
# o = F.scaled_dot_product_attention(
#     x[1, 1], x[1, 1], x[1, 1], attn_mask=attn_mask[1]
# )
# print(o)

# # print(
# #     F.scaled_dot_product_attention(
# #         x[0, 0], x[0, 0], x[0, 0], attn_mask=attn_mask
# #     )
# # )
