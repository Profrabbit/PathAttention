import torch
import numpy as np

line = [1, 2, 3]
a = torch.tensor(np.array(line[1:]).astype(np.float))
b = torch.tensor(np.append(np.array(line[1:]), np.array([0] * 2)).astype(np.float))
print(b)
