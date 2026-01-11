# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week1/2-10_Logistic_Regression.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 2]
import matplotlib.pyplot as plt
import numpy as np

# %% [code cell 3]
w = np.arange(-10, 10, 0.001)
b = 1
x = np.array([1.3, 1.2, 3.5, 4.1])
y = np.array([0, 0, 1, 1])
cost = []
for i in range(len(w)):
    cost.append(sum((w[i] * x + b - y) ** 2) / len(x))

# %% [code cell 4]
plt.plot(w, cost, 'ro', markersize=0.1)
plt.axis([-12, 15, -100, 1000])
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()

# %% [code cell 5]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
cost = []
for i in range(len(w)):
    cost.append(sum((sigmoid(w[i] * x + b) - y) ** 2) / len(x))

# %% [code cell 6]
plt.plot(w, cost, 'ro', markersize=0.1)
plt.axis([-7.5, 7.5, 0.1, 0.7])
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()

# %% [code cell 7]
cost = []
for i in range(len(w)):
    cost.append(sum(-y*np.log(sigmoid(w[i]*x+b))-(1-y)*np.log(1-sigmoid(w[i]*x+b))))

# %% [code cell 8]
plt.plot(w, cost, 'ro', markersize=0.1)
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()
