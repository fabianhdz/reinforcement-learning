# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week1/2-8_Linear_Classifier.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 2]
import matplotlib.pyplot as plt
import numpy as np

train_x = np.array([2, 3, 4, 5, 6, -2, -3, -4, -5, -6])
train_y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])

# %% [code cell 3]
plt.plot(train_x[0:5], train_y[0:5], 'bo', markersize=3)
plt.plot(train_x[5:], train_y[5:], 'ro', markersize=3)
plt.axis([-10, 10, -2, 2])

# %% [code cell 4]
w, b = 0, 0
alpha = 0.05

for i in range(10):
    w = w - alpha * (1/len(train_x)) * sum((w * train_x + b - train_y) * train_x)
    b = b - alpha * (1/len(train_x)) * sum(w * train_x + b - train_y)
print("w = %f, b = %f" % (w, b))

# %% [code cell 5]
plt.plot(train_x[0:5], train_y[0:5], 'bo', markersize=3)
plt.plot(train_x[5:], train_y[5:], 'ro', markersize=3)
plt.axis([-10, 10, -2, 2])
xl = np.linspace(-10, 10, 100)
plt.plot(xl, w * xl + b)
plt.grid(True)
plt.show()

# %% [code cell 7]
import matplotlib.pyplot as plt
import numpy as np

def hypothesis(X, w, b):
    return np.dot(X, w) + b

train_X = np.array([[1.5, 2.7, 1.3],
                    [2.4, 1.7, 2.1],
                    [2.5, 1.3, 2.2],
                    [8.5, 5.3, 4.8],
                    [4.9, 6.4, 5.7],
                    [7.2, 7.1, 7.4]])
train_y = np.array([1, 1, 1, -1, -1, -1])

# %% [code cell 8]
test_X = np.array([[2.4, 2.5, 0.7],
                    [5.9, 4.4, 5.2],
                    [0.2, 0.5, 0.6],
                    [4.3, 4.5, 5.5]])
test_y = np.array([1, -1, 1, -1])

# %% [code cell 9]
w = np.zeros(np.size(train_X, 1))
b = 0
alpha = 0.01

for i in range(2000):
    w = w - alpha * (1 / len(train_X)) * np.dot(np.transpose(np.dot(train_X, w) + b - train_y), train_X)
    b = b - alpha * (1 / len(train_X)) * sum(np.dot(train_X, w) + b - train_y)
print("w = ", w)
print("b = ", b)
