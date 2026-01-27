# -*- coding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt
import random

def f(x):
    return 1.45 * x + 72.4

rand_data = []
for i in range(5, 35):
    x = i + (random.random()-0.5) * 2
    y = f(x) + (random.random()-0.5) * 20
    rand_data.append([x, y])

# %%
data = np.array(rand_data)
x = data[:, 0]
y = data[:, 1]
plt.scatter(x, y)
w, b= 0, 0
alpha = 0.003
xl = np.linspace(0, 40, 100)
for i in range(20001):
  w = w - alpha * (1/len(data)) * sum((w * x + b - y) * x)
  b = b - alpha * (1/len(data)) * sum((w * x + b - y))
  if i % 1000 == 0:
    print("w = %.5f b = %.5f" % (w, b))
    print("mse = %.5f" % (sum(((w*x+b)-y)**2)/len(data)))
plt.plot(xl, w * xl + b)
plt.show()
