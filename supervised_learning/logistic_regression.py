# -*- coding: utf-8 -*-

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
w = np.arange(-10, 10, 0.001)
b = 1
x = np.array([1.3, 1.2, 3.5, 4.1])
y = np.array([0, 0, 1, 1])
cost = []
for i in range(len(w)):
    cost.append(sum((w[i] * x + b - y) ** 2) / len(x))

# %%
plt.plot(w, cost, 'ro', markersize=0.1)
plt.axis([-12, 15, -100, 1000])
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
cost = []
for i in range(len(w)):
    cost.append(sum((sigmoid(w[i] * x + b) - y) ** 2) / len(x))

# %%
plt.plot(w, cost, 'ro', markersize=0.1)
plt.axis([-7.5, 7.5, 0.1, 0.7])
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()

# %%
cost = []
for i in range(len(w)):
    cost.append(sum(-y*np.log(sigmoid(w[i]*x+b))-(1-y)*np.log(1-sigmoid(w[i]*x+b))))

# %%
plt.plot(w, cost, 'ro', markersize=0.1)
plt.xlabel("w")
plt.ylabel("Cost(w)")
plt.show()
