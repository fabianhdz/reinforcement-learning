"""Load image and CSV datasets through PyTorch data loaders."""

import pandas as pd
import torch
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
        x = torch.tensor(self.data.iloc[index, 0:4].to_numpy(), dtype=float)
        y = torch.tensor(self.data.iloc[index, 4], dtype=float)
        return x, y


def print_batches(dataloader, *, max_batches=None):
    for index, (features, targets) in enumerate(dataloader):
        if max_batches is not None and index >= max_batches:
            break
        print("x =", features)
        print("y =", targets)


def show_mnist_batches(*, root="./data/mnist", batch_size=5, max_batches=3):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)),
    ])
    train_data = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    print_batches(train_loader, max_batches=max_batches)
    return train_loader, test_loader


def show_csv_batches(csv_file=IRIS_CSV_URL, *, batch_size=5, max_batches=None):
    loader = DataLoader(CustomDataset(csv_file), batch_size=batch_size, shuffle=True)
    print_batches(loader, max_batches=max_batches)
    return loader


def main():
    show_mnist_batches()
    show_csv_batches()

if __name__ == "__main__":
    main()
