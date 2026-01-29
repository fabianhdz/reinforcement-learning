"""Fit and plot a linear regression model with gradient descent."""

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


def plot_fit(x, y, slope, intercept):
    fitted_x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(fitted_x, slope * fitted_x + intercept, label="Prediction")
    ax.set(title="Linear regression", xlabel="x", ylabel="y")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    data = np.array([
        [2, 81],
        [4, 93],
        [6, 91],
        [8, 97],
    ])
    x, y = data[:, 0], data[:, 1]
    slope, intercept = fit_linear_regression(x, y)
    print("w = %f, b = %f" % (slope, intercept))
    plot_fit(x, y, slope, intercept)
    plt.show()

if __name__ == "__main__":
    main()
