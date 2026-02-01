"""Dense and convolutional FashionMNIST classifiers."""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LABELS = ("T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
          "Shirt", "Sneaker", "Bag", "Ankle Boot")


class FashionDNN1(nn.Module):
    def __init__(self):
        super(FashionDNN1, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
    def forward(self, input_data):
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class FashionDNN2(nn.Module):
    def __init__(self):
        super(FashionDNN2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=784, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self, input_data):
        out = input_data.view(-1, 784)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


class FashionDNN3(nn.Module):
    def __init__(self):
        super(FashionDNN3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=64*6*6, out_features=600),
            nn.Linear(600, 120),
            nn.Linear(120, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.layer3(x)
        return x


def create_dataloaders(*, root="./data", batch_size=1000):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.FashionMNIST(root, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=False),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
    )


def plot_samples(dataset, *, count=20):
    samples = [dataset[np.random.randint(len(dataset))] for _ in range(count)]
    columns = 4
    rows = (count + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=(8, 8))
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < len(samples):
            image, label = samples[index]
            ax.imshow(image.squeeze().numpy(), cmap="gray")
            ax.set_title(LABELS[label])
    fig.tight_layout()
    return fig


def train_model(model, dataloader, *, epochs=10, learning_rate=0.01):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = samples = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            samples += targets.size(0)
        metrics = {"loss": total_loss / len(dataloader), "accuracy": correct / samples}
        history.append(metrics)
        print(f"Epoch {epoch}/{epochs}, loss: {metrics['loss']:.4f}, "
              f"accuracy: {metrics['accuracy']:.4f}")
    return history


def evaluate_model(model, dataloader):
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct = samples = 0
    try:
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs).argmax(dim=1)
                correct += (predictions == targets).sum().item()
                samples += targets.size(0)
    finally:
        model.train(was_training)
    return correct / samples


def main():
    train_loader, test_loader = create_dataloaders()
    plot_samples(train_loader.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FashionDNN3().to(device)
    train_model(model, train_loader)
    print(f"Test accuracy: {evaluate_model(model, test_loader):.4f}")
    plt.show()

if __name__ == "__main__":
    main()
