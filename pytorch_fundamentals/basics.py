# -*- coding: utf-8 -*-

# %%
import torch
print(torch.__version__)
print(torch.tensor([[1, 2], [3, 4]])) # Create a 2D tensor
print(torch.tensor([[1, 2], [3, 4]], device='cuda:0')) # Generating a tensor on the GPU
print(torch.tensor([[1, 2], [3, 4]], dtype=torch.float64)) # Create a tensor using dtype

# %%
temp = torch.tensor([[1, 2],[3, 4]])
print(temp.numpy()) # Convert tensor to ndarray
temp = torch.tensor([[1, 2],[3, 4]], device='cuda:0')
print(temp.to('cpu').numpy()) # Convert tensor on GPU to tensor on CPU, and then convert to ndarray

# %%
t = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7])
print(t[0], t[1], t[-1])
print(t[2:5], t[4:])

# %%
t1 = torch.tensor([1, 2, 3])
t2 = torch.tensor([4, 5, 6])
t3 = t1 + t2
print(t3)

# %%
# reshaping tensor
tensor = torch.arange(10)
print(tensor)
reshaped = tensor.view(2, 5)
print(reshaped)
print('\n')

# stacking tensors
tensor1 = torch.Tensor([1,2,3])
tensor2 = torch.Tensor([4,5,6])
stacked = torch.stack((tensor1, tensor2))
print(stacked)
print('\n')

# concatenating tensors
cat = torch.cat((tensor1, tensor2))
print(cat)
print('\n')

# transposing tensor
tensor = torch.Tensor([[1,2,3],[4,5,6]])
transposed = tensor.t()
print(transposed)
