import torch


x = torch.tensor([-2, -1, 0, 0, 0]).reshape(1, 5)
y = torch.tensor([-2, -1, 0, 0, 0]).reshape(5, 1)
z = x + y

print(z)
