"""Train a multiclass linear classifier with one-hot targets."""

import numpy as np


def example_data():
    trainX = np.array([
        [0.1, 0.2, 0.3, 0.2],
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
        [13.1, 10.2, 12.4, 11.7],
    ])
    trainY = np.array([
        [1, 0, 0],
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
        [0, 0, 1],
    ])
    testX = np.array([
        [0.5, 0.4, 0.6, 0.5],
        [5.4, 5.6, 5.5, 5.2],
        [11.7, 11.6, 11.5, 11.4],
    ])
    testY = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])
    return trainX, trainY, testX, testY


def train_classifier(x, targets, *, learning_rate=0.005, steps=1000):
    weights = np.zeros((x.shape[1], targets.shape[1]))
    biases = np.zeros(targets.shape[1])
    for _ in range(steps):
        for column in range(targets.shape[1]):
            error = x @ weights[:, column] + biases[column] - targets[:, column]
            weights[:, column] -= learning_rate * (error @ x) / len(x)
            error = x @ weights[:, column] + biases[column] - targets[:, column]
            biases[column] -= learning_rate * np.mean(error)
    return weights, biases


def predict(x, weights, biases):
    return np.argmax(x @ weights + biases, axis=1)


def main():
    train_x, train_y, test_x, test_y = example_data()
    weights, biases = train_classifier(train_x, train_y)
    print("W:", weights)
    print("B:", biases)
    print("accuracy:", np.mean(predict(test_x, weights, biases) == test_y.argmax(axis=1)))

if __name__ == "__main__":
    main()
