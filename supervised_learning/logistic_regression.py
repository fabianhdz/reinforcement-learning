"""Compare regression and classification cost curves."""

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cost_curves(weights, x, y, bias=1):
    logits = np.asarray(weights)[:, None] * np.asarray(x) + bias
    probabilities = sigmoid(logits)
    return {
        "Linear squared error": np.mean((logits - y) ** 2, axis=1),
        "Sigmoid squared error": np.mean((probabilities - y) ** 2, axis=1),
        "Cross-entropy": np.sum(np.logaddexp(0, logits) - y * logits, axis=1),
    }


def plot_costs(weights, costs):
    limits = [((-12, 15), (-100, 1000)), ((-7.5, 7.5), (0.1, 0.7)), (None, None)]
    figures = []
    for (title, values), (xlim, ylim) in zip(costs.items(), limits):
        fig, ax = plt.subplots()
        ax.plot(weights, values, "ro", markersize=0.1)
        ax.set(title=title, xlabel="w", ylabel="Cost(w)")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        fig.tight_layout()
        figures.append(fig)
    return figures


def main():
    weights = np.arange(-10, 10, 0.001)
    x = np.array([1.3, 1.2, 3.5, 4.1])
    y = np.array([0, 0, 1, 1])
    plot_costs(weights, cost_curves(weights, x, y))
    plt.show()

if __name__ == "__main__":
    main()
