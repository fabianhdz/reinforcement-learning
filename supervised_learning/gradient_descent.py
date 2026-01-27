# -*- coding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt

data = np.array([[2, 81], [4, 93], [6, 91], [8, 97]])
x = data[:, 0]
y = data[:, 1]
w, b= 0, 0
alpha = 0.05

plt.scatter(x, y)
xl = np.linspace(0, 10, 100)

for i in range(2000):
  w = w - alpha * (1/len(data)) * sum((w * x + b - y) * x)
  b = b - alpha * (1/len(data)) * sum((w * x + b - y))

print("w = %f, b = %f" % (w, b))
plt.plot(xl, w * xl + b)
plt.show()
