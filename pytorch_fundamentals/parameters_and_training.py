"""Train a small classifier on a CSV dataset."""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


IRIS_CSV_URL = "https://raw.githubusercontent.com/dkims/CSCI4341/main/iris.csv"


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        # initialization code
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        # return the size of the dataset
        return len(self.data)

    def __getitem__(self, index):
        # return a sample from the dataset at a given index
        x = torch.tensor(self.data.iloc[index, 0:4].to_numpy(), dtype=torch.float32)
        y = torch.tensor(self.data.iloc[index, 4], dtype=torch.long)
        return x, y


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


def create_dataloader(csv_file=IRIS_CSV_URL, *, batch_size=150):
    return DataLoader(CustomDataset(csv_file), batch_size=batch_size, shuffle=True)


def train_model(model, dataloader, *, epochs=200, learning_rate=0.05):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        for features, targets in dataloader:
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            print(loss)
            print((predictions.argmax(dim=1) == targets).sum())
    return model


def main():
    model = Model()
    train_model(model, create_dataloader())

if __name__ == "__main__":
    main()
