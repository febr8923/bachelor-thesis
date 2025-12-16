import torch
import numpy as np

a = torch.zeros(5)
b = a.numpy()
print(type(b))

c = np.ones(5)
d = torch.from_numpy(c)
print(type(d))
