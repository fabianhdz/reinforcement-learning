"""Convolutional MNIST classification."""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
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


def create_dataloaders(*, root="./data/mnist", batch_size=200):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=False),
    )


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
    model = Model()
    train_model(model, train_loader)
    print(f"Test accuracy: {evaluate_model(model, test_loader):.4f}")

if __name__ == "__main__":
    main()
