# -*- coding: utf-8 -*-

# %%
import numpy as np
a_and_b = np.array([3.4, 77]) # a and b
data = np.array([[2, 81], [4, 93], [6, 91], [8, 97]])
train_x = data[:, 0]
train_y = data[:, 1]

# %%
print(a_and_b)

# %%
print(data)

# %%
print(   a_and_b[0] * train_x + a_and_b[1]      )

# %%
mse = sum(((a_and_b[0] * train_x + a_and_b[1]) - train_y)**2)/len(train_x)
print(mse)
