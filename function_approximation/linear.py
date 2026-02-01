"""Fit a linear model to noisy samples."""

import random

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


def f(x):
    return 1.45 * x + 72.4


def generate_data(seed=None):
    rng = random.Random(seed)
    samples = []
    for value in range(5, 35):
        x = value + (rng.random() - 0.5) * 2
        y = f(x) + (rng.random() - 0.5) * 20
        samples.append([x, y])
    return np.array(samples)


def plot_fit(x, y, slope, intercept):
    fitted_x = np.linspace(0, 40, 100)
    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Data")
    ax.plot(fitted_x, slope * fitted_x + intercept, label="Prediction")
    ax.set(title="Function approximation", xlabel="x", ylabel="y")
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    data = generate_data()
    x, y = data[:, 0], data[:, 1]
    slope, intercept = fit_linear_regression(
        x, y, learning_rate=0.003, steps=20001, log_every=1000,
    )
    plot_fit(x, y, slope, intercept)
    plt.show()

if __name__ == "__main__":
    main()
