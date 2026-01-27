# -*- coding: utf-8 -*-

# %%
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
        x = torch.tensor(self.data.iloc[index, 0:4], dtype=torch.float32)
        y = torch.tensor(self.data.iloc[index, 4], dtype=torch.long)
        return x, y

url = 'https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv'
tensor_dataset = CustomDataset(url)
dataloader = DataLoader(tensor_dataset, batch_size=150, shuffle=True)

import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(4, 3)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = Model()

from torch.optim import optimizer

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.99 ** epoch)

# train
for epoch in range(1, 200+1):
    for x, y in dataloader:
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        #scheduler.step()

    if epoch % 20 == 0:
        print(loss)
        print((torch.max(pred, 1)[1] == y).sum()) # evaluation with train data
