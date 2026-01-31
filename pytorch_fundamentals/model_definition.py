"""Two ways to define a small feedforward network."""

import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_models():
    return Model(), Model2()


def main():
    for model in create_models():
        print(model)

if __name__ == "__main__":
    main()
