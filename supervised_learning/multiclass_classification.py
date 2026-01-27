# -*- coding: utf-8 -*-

# %%
import numpy as np

trainX = np.array([[0.1, 0.2, 0.3, 0.2],
                   [0.5, 0.4, 0.3, 0.7],
                   [0.3, 0.7, 0.4, 0.1],
                   [0.2, 0.8, 0.9, 0.3],
                   [1.1, 0.5, 0.2, 0.9],
                   [4.3, 5.3, 4.7, 4.2],
                   [4.5, 5.1, 5.3, 4.4],
                   [5.1, 4.8, 5.1, 4.6],
                   [4.9, 4.6, 4.9, 4.3],
                   [5.4, 5.5, 4.3, 4.7],
                   [10.1, 10.2, 10.3, 11.3],
                   [11.3, 11.2, 11.1, 10.3],
                   [12.5, 12.3, 12.1, 11.4],
                   [11.7, 11.8, 11.2, 12.8],
                   [13.1, 10.2, 12.4, 11.7]])
trainY = np.array([[1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1]])

# %%
testX = np.array([[0.5, 0.4, 0.6, 0.5],
                  [5.4, 5.6, 5.5, 5.2],
                  [11.7, 11.6, 11.5, 11.4]])
testY = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

# %%
W = np.zeros((4, 3))
B = np.zeros(3)
alpha = 0.005
N = trainX.shape[0]
c = W.shape[1]

# %%
for i in range(1000):
    for j in range(c):
        W[:, j] = W[:, j] - alpha * (1/N) * np.dot(np.transpose(np.dot(trainX, W[:, j])+B[j] - trainY[:, j]), trainX)
        B[j] = B[j] - alpha * (1/N) * sum(np.dot(trainX, W[:, j])+B[j] - trainY[:, j])
print('W:', W)
print('B:', B)
print('accuracy:', sum(np.argmax(np.dot(testX, W)+B, axis=1) == np.argmax(testY, axis=1))/testX.shape[0])
