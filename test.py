import torch

a = torch.nn.Linear(2, 3)
b = torch.nn.Linear(3, 2)
b.weight.data = a.weight.T
print(a.weight, b.weight)
a.weight.data = torch.tensor([[-0.5218, -0.2065,  0.3984],
        [-0.2393, -0.1471,  0.0788]], requires_grad=True)
print(a.weight, b.weight)
