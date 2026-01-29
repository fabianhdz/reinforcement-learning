"""One-dimensional and multivariable linear classification."""

import matplotlib.pyplot as plt
import numpy as np


def fit_linear_regression(x, y, *, learning_rate=0.05, steps=2000, log_every=0):
    x, y = np.asarray(x), np.asarray(y)
    slope = intercept = 0.0
    for step in range(steps):
        slope -= learning_rate * np.mean((slope * x + intercept - y) * x)
        intercept -= learning_rate * np.mean(slope * x + intercept - y)
        if log_every and step % log_every == 0:
            mse = np.mean((slope * x + intercept - y) ** 2)
            print(f"w = {slope:.5f} b = {intercept:.5f}")
            print(f"mse = {mse:.5f}")
    return slope, intercept


def fit_vectorized(x, y, *, learning_rate=0.01, steps=50000):
    weights = np.zeros(x.shape[1])
    bias = 0.0
    for _ in range(steps):
        weights -= learning_rate * ((x @ weights + bias - y) @ x) / len(x)
        bias -= learning_rate * np.mean(x @ weights + bias - y)
    return weights, bias


def hypothesis(x, weights, bias):
    return np.dot(x, weights) + bias


def example_data():
    train_X = np.array([
        [1.5, 2.7, 1.3],
        [2.4, 1.7, 2.1],
        [2.5, 1.3, 2.2],
        [8.5, 5.3, 4.8],
        [4.9, 6.4, 5.7],
        [7.2, 7.1, 7.4],
    ])
    train_y = np.array([1, 1, 1, -1, -1, -1])
    test_X = np.array([
        [2.4, 2.5, 0.7],
        [5.9, 4.4, 5.2],
        [0.2, 0.5, 0.6],
        [4.3, 4.5, 5.5],
    ])
    test_y = np.array([1, -1, 1, -1])
    return train_X, train_y, test_X, test_y


def plot_classifier(x, y, slope, intercept):
    fitted_x = np.linspace(-10, 10, 100)
    fig, ax = plt.subplots()
    ax.plot(x[y == 1], y[y == 1], "bo", markersize=3)
    ax.plot(x[y == -1], y[y == -1], "ro", markersize=3)
    ax.plot(fitted_x, slope * fitted_x + intercept, label="Model")
    ax.set(title="Linear classifier", xlabel="x", ylabel="Class score",
           xlim=(-10, 10), ylim=(-2, 2))
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    return fig


def run_one_dimensional():
    x = np.array([2, 3, 4, 5, 6, -2, -3, -4, -5, -6])
    y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    slope, intercept = fit_linear_regression(x, y, steps=10)
    print("w = %f, b = %f" % (slope, intercept))
    plot_classifier(x, y, slope, intercept)
    return slope, intercept


def run_multivariable():
    train_x, train_y, test_x, test_y = example_data()
    weights, bias = fit_vectorized(train_x, train_y, steps=2000)
    print("w =", weights)
    print("b =", bias)
    return weights, bias


def main():
    run_one_dimensional()
    run_multivariable()
    plt.show()

if __name__ == "__main__":
    main()
