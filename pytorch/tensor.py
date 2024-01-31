import torch

x = torch.empty(5, 3, 2)
print(x)

x = torch.rand(5, 3, 2)
print(x)

x = torch.zeros(2, 3)
print(x)

x = torch.ones(4, 2, dtype=torch.float16)
print(x)
print(x.size(), x.dim())

x = torch.zeros(2, 2)
y = torch.ones(2, 2)
z = torch.add(x, y)
print(z)