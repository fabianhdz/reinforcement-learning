# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week2/4-5-Pytorch_Define_Model.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 4]
import torch.nn as nn

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

model = Model()

# %% [code cell 5]
import torch.nn.functional as F

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model2 = Model2()

