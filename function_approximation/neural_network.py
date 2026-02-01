"""Approximate a noisy nonlinear function with a neural network."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def true_fun(X):
    noise = np.random.rand(X.shape[0]) * 0.4 - 0.2
    return np.cos(1.5 * np.pi * X) + X + noise


def train_model(model, *, steps=10000, learning_rate=0.001):
    data_x = np.random.rand(10000) * 5
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        batch_x = np.random.choice(data_x, 32)
        inputs = torch.from_numpy(batch_x).float().unsqueeze(1)
        targets = torch.from_numpy(true_fun(batch_x)).float().unsqueeze(1)
        loss = F.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
    return model


def plot_results(model):
    x = np.linspace(0, 5, 100)
    with torch.no_grad():
        predictions = model(torch.from_numpy(x).float().unsqueeze(1)).numpy().ravel()
    fig, ax = plt.subplots()
    ax.plot(x, true_fun(x), label="Truth")
    ax.plot(x, predictions, label="Prediction")
    ax.set(xlim=(-0.2, 5.5), ylim=(-1, 6))
    ax.legend(loc="lower right", fontsize=15)
    ax.grid(True)
    fig.tight_layout()
    return fig


def main():
    model = train_model(Model())
    plot_results(model)
    plt.show()

if __name__ == "__main__":
    main()
