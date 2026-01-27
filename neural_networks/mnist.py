# -*- coding: utf-8 -*-

# %%
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
batch_size = 200
dataset_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %%
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

model = Model()


# %%
from torch.optim import optimizer
import torch
import numpy as np

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

# train
n_epoch = 3
for epoch in range(1, n_epoch+1):
    train_losses = []
    train_acc = []
    for idx, (x, y) in enumerate(dataset_train):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        train_losses.append(loss.item())
        train_acc.append( (torch.max(pred, 1)[1] == y).sum()/batch_size )
        loss.backward()
        optimizer.step()
        #scheduler.step()
    print('epoch: {}, loss:{}, acc:{}'.format(epoch, sum(train_losses)/(idx+1), sum(train_acc)/(idx+1) ))

# %%
dataset_test = DataLoader(test_dataset, batch_size=batch_size)
test_acc = []
for i, (x, y) in enumerate(dataset_test):
    test_pred = model(x)
    test_acc.append((torch.max(test_pred, 1)[1] == y).sum()/batch_size)
print('test acc:{}'.format(sum(test_acc)/(len(test_acc) )) )
