"""Regression, data splitting, and classification examples."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split


class LinearRegressionModel(nn.Module):
    def __init__(self, input_features=1):
        super().__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        return self.linear(x)


def plot_study_scores(hours, scores, predictions):
    fig, ax = plt.subplots()
    ax.plot(hours, scores, "ro", label="Actual")
    ax.plot(hours, predictions, "b-", label="Predicted")
    ax.set(xlabel="Study Hours", ylabel="Scores")
    ax.legend()
    fig.tight_layout()
    return fig


def train_regression(features, targets, *, learning_rate, epochs=1000):
    features = torch.as_tensor(features, dtype=torch.float32)
    targets = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)
    model = LinearRegressionModel(features.shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    with torch.no_grad():
        predictions = model(features).numpy()
    print(predictions)
    return model, predictions


def run_house_price_regression():
    features = [[1200.0, 3], [1500.0, 4], [1800.0, 3], [2000.0, 5]]
    targets = [300000.0, 400000.0, 450000.0, 500000.0]
    return train_regression(features, targets, learning_rate=1e-8)


def run_data_split():
    X = [[1200.0, 3], [1500.0, 4], [1800.0, 3], [2000.0, 5], [2200.0, 4], [2400.0, 5], [2600.0, 4], [2800.0, 5], [2400.0, 5], [2600.0, 4]]
    y = [300000.0, 400000.0, 450000.0, 500000.0, 550000.0, 600000.0, 650000.0, 700000.0, 600000.0, 650000.0]
    train_x, temporary_x, train_y, temporary_y = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    val_x, test_x, val_y, test_y = train_test_split(
        temporary_x, temporary_y, test_size=0.33, random_state=42,
    )
    print(f"Training set size: {len(train_x)}")
    print(f"Validation set size: {len(val_x)}")
    print(f"Test set size: {len(test_x)}")
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def run_iris_classification():
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=7,
    )
    model = LogisticRegression(max_iter=200).fit(train_x, train_y)
    predictions = model.predict(test_x)
    metrics = {
        "Accuracy": accuracy_score(test_y, predictions),
        "Precision": precision_score(test_y, predictions, average="macro"),
        "Recall": recall_score(test_y, predictions, average="macro"),
        "F1 Score": f1_score(test_y, predictions, average="macro"),
    }
    for label, value in metrics.items():
        print(f"{label}: {value:.2f}")
    print("Classification Report:\n", classification_report(test_y, predictions))
    return model, metrics


def run_study_score_regression():
    rng = np.random.RandomState(6370)
    hours = np.arange(1, 11)
    scores = np.arange(50, 100, 5) + rng.normal(0, 5, len(hours))
    data = pd.DataFrame({"study_hours": hours, "scores": scores})
    print(data)
    model, predictions = train_regression(data[["study_hours"]].values, scores,
                                           learning_rate=0.01)
    plot_study_scores(hours, scores, predictions.ravel())
    return model


def main():
    run_house_price_regression()
    run_data_split()
    run_iris_classification()
    run_study_score_regression()
    plt.show()

if __name__ == "__main__":
    main()
