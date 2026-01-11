# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week1/2-2_linear_regression.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 2]
import numpy as np
a_and_b = np.array([3.4, 77]) # a and b
data = np.array([[2, 81], [4, 93], [6, 91], [8, 97]])
train_x = data[:, 0]
train_y = data[:, 1]

# %% [code cell 3]
print(a_and_b)

# %% [code cell 4]
print(data)

# %% [code cell 5]
print(   a_and_b[0] * train_x + a_and_b[1]      )

# %% [code cell 6]
mse = sum(((a_and_b[0] * train_x + a_and_b[1]) - train_y)**2)/len(train_x)
print(mse)
