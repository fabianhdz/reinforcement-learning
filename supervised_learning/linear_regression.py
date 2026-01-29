"""Evaluate a linear regression model and its mean squared error."""

import numpy as np


def predict(x, slope, intercept):
    return slope * np.asarray(x) + intercept


def mean_squared_error(actual, predicted):
    return np.mean((np.asarray(predicted) - actual) ** 2)


def main():
    parameters = np.array([3.4, 77])
    data = np.array([
        [2, 81],
        [4, 93],
        [6, 91],
        [8, 97],
    ])
    predictions = predict(data[:, 0], *parameters)
    print(parameters)
    print(data)
    print(predictions)
    print(mean_squared_error(data[:, 1], predictions))

if __name__ == "__main__":
    main()
