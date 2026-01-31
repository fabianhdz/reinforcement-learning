"""A feedforward network with dropout."""

import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.dropout = nn.Dropout(0.5)  # Dropout layer with dropout probability of 0.5
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)  # Apply dropout to the output of fc1
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def main():
    print(MyModel())

if __name__ == "__main__":
    main()
