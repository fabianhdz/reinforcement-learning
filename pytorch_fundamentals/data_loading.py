# -*- coding: utf-8 -*-
# Source notebook: Reinforcement Learning/Week2/4-4-Pytorch_Data_Loading.ipynb
# Code cells are preserved in notebook order; markdown and outputs are omitted.
# Notebook shell commands are retained as comments for use from a terminal.

# %% [code cell 5]
import torchvision.transforms as transforms

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (1.0,))
])

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import requests

download_root = './data/mnist'
train_dataset = MNIST(download_root, transform=mnist_transform, train=True, download=True)
test_dataset = MNIST(download_root, transform=mnist_transform, train=False, download=True)
dataset = DataLoader(train_dataset, batch_size=5, shuffle=True)
for x, y in dataset:
    print('x =', x)
    print('y =', y)

# %% [code cell 9]
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # initialization code
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # return the size of the dataset
        return len(self.data)

    def __getitem__(self, index):
        # return a sample from the dataset at a given index
        x = torch.tensor(self.data.iloc[index, 0:4], dtype=float)
        y = torch.tensor(self.data.iloc[index, 4], dtype=float)
        return x, y

url = 'https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv'
tensor_dataset = CustomDataset(url)
dataset = DataLoader(tensor_dataset, batch_size=5, shuffle=True)
for x, y in dataset:
    print('x =', x)
    print('y =', y)
