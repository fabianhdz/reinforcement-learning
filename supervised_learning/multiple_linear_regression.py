"""Compare sequential and vectorized multivariable regression updates."""

import numpy as np


def example_data(variant=1):
    if variant == 0:
        return np.array([
            [3.9, 2.1, 7.5, 19.05],
            [2.1, 3.2, 4.5, 13.25],
            [4.7, 4.1, 7.8, 22.05],
            [8.5, 5.4, 1.2, 12.05],
            [1.9, 3.4, 6.2, 16.75],
            [6.3, 4.3, 7.7, 22.85],
            [8.9, 4.4, 3.3, 15.45],
        ])
    return np.array([
        [3.5, 4.7, 2.3, 20.8],
        [4.4, 5.7, 4.1, 29.1],
        [2.5, 7.3, 1.2, 21.7],
        [8.5, 3.3, 4.8, 30.5],
        [4.9, 6.4, 5.7, 35.8],
        [7.2, 7.1, 7.4, 44.6],
        [5.6, 8.2, 6.5, 42.5],
    ])


def fit_sequential(x, y, *, learning_rate=0.01, steps=10000):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        for column in range(x.shape[1]):
            weights[column] -= learning_rate * np.mean((x @ weights + bias - y) * x[:, column])
        bias -= learning_rate * np.mean(x @ weights + bias - y)
    return weights, bias


def fit_vectorized(x, y, *, learning_rate=0.01, steps=50000):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        weights -= learning_rate * ((x @ weights + bias - y) @ x) / len(x)
        bias -= learning_rate * np.mean(x @ weights + bias - y)
    return weights, bias


def main():
    data = example_data()
    x, y = data[:, :-1], data[:, -1]
    print(data.shape)
    weights, bias = fit_sequential(x, y)
    print("w1 = %f, w2 = %f, w3 = %f, b = %f" % (*weights, bias))
    weights, bias = fit_vectorized(x, y)
    print("w =", weights, "b =", bias)

if __name__ == "__main__":
    main()
