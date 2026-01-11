# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week1/2-5_Multiple_Linear_Regression.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 3]
import numpy as np
data = np.array([[3.9, 2.1, 7.5, 19.05],
                 [2.1, 3.2, 4.5, 13.25],
                 [4.7, 4.1, 7.8, 22.05],
                 [8.5, 5.4, 1.2, 12.05],
                 [1.9, 3.4, 6.2, 16.75],
                 [6.3, 4.3, 7.7, 22.85],
                 [8.9, 4.4, 3.3, 15.45]])

# %% [code cell 4]
data = np.array([[3.5, 4.7, 2.3, 20.8],
                 [4.4, 5.7, 4.1, 29.1],
                 [2.5, 7.3, 1.2, 21.7],
                 [8.5, 3.3, 4.8, 30.5],
                 [4.9, 6.4, 5.7, 35.8],
                 [7.2, 7.1, 7.4, 44.6],
                 [5.6, 8.2, 6.5, 42.5]]) # Lab 10

# %% [code cell 5]
print(data.shape)
x = data[:, :-1]
y = data[:, -1]

# %% [code cell 6]
w1, w2, w3, b = 0, 0, 0, 0
alpha = 0.01

# %% [code cell 7]
for i in range(10000):
    w1 = w1 - alpha * (1/len(data)) * sum((x[:, 0] * w1 + x[:, 1] * w2 + x[:, 2] * w3 + b - y ) * x[:, 0])
    w2 = w2 - alpha * (1/len(data)) * sum((x[:, 0] * w1 + x[:, 1] * w2 + x[:, 2] * w3 + b - y ) * x[:, 1])
    w3 = w3 - alpha * (1/len(data)) * sum((x[:, 0] * w1 + x[:, 1] * w2 + x[:, 2] * w3 + b - y ) * x[:, 2])
    b  =  b - alpha * (1/len(data)) * sum( x[:, 0] * w1 + x[:, 1] * w2 + x[:, 2] * w3 + b - y )
print("w1 = %f, w2 = %f, w3 = %f, b = %f" % (w1, w2, w3, b))

# %% [code cell 8]
w = np.array([0, 0, 0])
b = 0
alpha = 0.01

# %% [code cell 9]
for i in range(50000):
    w = w - alpha * (1/len(data)) * np.dot(np.transpose(np.dot(x, w) + b - y), x)
    b  =  b - alpha * (1/len(data)) * sum(np.dot(x, w) + b -y)
print("w =", w, "b =", b)
