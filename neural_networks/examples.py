"""Independent neural network, tensor, and activation function examples."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # From input layer to first hidden layer (flattened 28x28 pixel image)
        self.fc2 = nn.Linear(128, 64)   # Second hidden layer
        self.fc3 = nn.Linear(64, 10)    # Output layer (classifying digits 0-9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # First hidden layer + ReLU activation
        x = torch.relu(self.fc2(x))  # Second hidden layer + ReLU activation
        x = self.fc3(x)              # Output layer (activation function applied in the loss function)
        return x


class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(2, 1)  # Single linear layer

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # Output layer with Sigmoid activation
        return x


class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # From input layer to hidden layer
        self.fc2 = nn.Linear(2, 1)  # From hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Hidden layer with Sigmoid activation
        x = torch.sigmoid(self.fc2(x))  # Output layer with Sigmoid activation
        return x


class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Hidden layer 1 + ReLU activation function
        x = torch.relu(self.fc2(x))  # Hidden layer 2 + ReLU activation function
        x = torch.softmax(self.fc3(x), dim=1)  # Output layer + Softmax activation function
        return x


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.linear(x)


class BatchNormalizedNN(nn.Module):
    def __init__(self):
        super(BatchNormalizedNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input image to a vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_classifier(model, dataloader, optimizer, criterion, *, epochs,
                     device=None, log_every=1):
    """Train a classifier and return loss and accuracy for each epoch."""
    if device is None:
        device = next(model.parameters()).device
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = correct = samples = batches = 0
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
            batches += 1
        if not samples:
            raise ValueError("The training loader is empty")
        metrics = {"loss": total_loss / batches, "accuracy": correct / samples}
        history.append(metrics)
        if log_every and epoch % log_every == 0:
            print(f"Epoch {epoch}/{epochs}, loss: {metrics['loss']:.4f}, "
                  f"accuracy: {metrics['accuracy']:.4f}")
    return history


def evaluate_classifier(model, dataloader, *, device=None, collect_errors=False):
    """Return accuracy and optionally misclassified images, predictions, and labels."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct = samples = 0
    errors = []
    try:
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs).argmax(dim=1)
                correct += (predictions == targets).sum().item()
                samples += targets.size(0)
                if collect_errors:
                    for index in (predictions != targets).nonzero(as_tuple=True)[0]:
                        errors.append((inputs[index].cpu(), predictions[index].item(),
                                       targets[index].item()))
    finally:
        model.train(was_training)
    if not samples:
        raise ValueError("The evaluation loader is empty")
    return correct / samples, errors


def plot_image_grid(images, titles=None, *, columns=3, cmap="gray", figsize=None):
    """Plot any number of images, hiding unused cells in the grid."""
    images = list(images)
    if not images:
        return None
    if columns < 1:
        raise ValueError("columns must be positive")
    if titles is not None and len(titles) != len(images):
        raise ValueError("Each image must have a title")
    rows = (len(images) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False, figsize=figsize)
    for index, ax in enumerate(axes.flat):
        ax.axis("off")
        if index < len(images):
            ax.imshow(images[index], cmap=cmap)
            if titles is not None:
                ax.set_title(titles[index])
    fig.tight_layout()
    return fig


def show_models():
    for model_type in (SimpleNN, ImageClassifier, BatchNormalizedNN):
        print(model_type())


def train_binary_model(model, features, targets, *, epochs, learning_rate, log_every):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(features), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % log_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    with torch.no_grad():
        predictions = model(features).round()
    print(f"Predicted ({type(model).__name__}):\n{predictions}")
    return predictions


def run_xor_comparison(*, single_epochs=10000, multilayer_epochs=50000):
    features = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    targets = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    single = SingleLayerPerceptron()
    multilayer = XOR_MLP()
    train_binary_model(single, features, targets, epochs=single_epochs,
                       learning_rate=0.1, log_every=1000)
    train_binary_model(multilayer, features, targets, epochs=multilayer_epochs,
                       learning_rate=0.05, log_every=5000)
    return single, multilayer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)


def plot_activation_functions():
    x = np.linspace(-10, 10, 100)
    curves = [
        ("Sigmoid Function", {"": sigmoid(x)}),
        ("Tanh Function", {"": tanh(x)}),
        ("ReLU and Leaky ReLU Functions", {"ReLU": relu(x), "Leaky ReLU": leaky_relu(x)}),
    ]
    figures = []
    for title, values in curves:
        fig, ax = plt.subplots()
        for label, outputs in values.items():
            ax.plot(x, outputs, label=label or None)
        ax.set(title=title, xlabel="Input", ylabel="Output")
        ax.grid(True)
        if ax.get_legend_handles_labels()[1]:
            ax.legend()
        fig.tight_layout()
        figures.append(fig)
    return figures


def create_iris_loaders():
    iris = load_iris()
    train_x, test_x, train_y, test_y = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    train_x = torch.tensor(scaler.fit_transform(train_x), dtype=torch.float32)
    test_x = torch.tensor(scaler.transform(test_x), dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.long)
    test_y = torch.tensor(test_y, dtype=torch.long)
    return (
        DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True),
        DataLoader(TensorDataset(test_x, test_y), batch_size=len(test_y)),
    )


def run_iris_classification(*, epochs=5000):
    train_loader, test_loader = create_iris_loaders()
    model = SimpleModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    train_classifier(model, train_loader, optimizer, nn.CrossEntropyLoss(),
                      epochs=epochs, log_every=100)
    accuracy, _ = evaluate_classifier(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return model


def demonstrate_tensors():
    print(torch.tensor([1, 2, 3]))
    print(torch.zeros(3, 3))
    print(torch.ones(2, 2))
    print(torch.randn(2, 2))
    first = torch.tensor([1, 2, 3])
    second = torch.tensor([4, 5, 6])
    print(first + second)
    print(first * second)
    if torch.cuda.is_available():
        print(first.to("cuda"))
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x + 2
    z = (y * y * 2).mean()
    z.backward()
    print(x.grad)


def create_mnist_loaders(*, root="./data"):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.RandomRotation(10), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),
    ])
    train_data = datasets.MNIST(root, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root, train=False, download=True, transform=transform)
    return (
        DataLoader(train_data, batch_size=64, shuffle=True),
        DataLoader(test_data, batch_size=1000, shuffle=False),
    )


def plot_samples(images, labels, *, count=6):
    count = min(count, len(images))
    return plot_image_grid([image.cpu().numpy().squeeze() for image in images[:count]],
                            [f"Label: {label.item()}" for label in labels[:count]])


def plot_misclassified(errors, *, count=6):
    samples = errors[:count]
    return plot_image_grid([image.numpy().squeeze() for image, _, _ in samples],
                            [f"Pred: {prediction}, True: {target}"
                             for _, prediction, target in samples])


def run_mnist_classification(*, epochs=5, root="./data"):
    train_loader, test_loader = create_mnist_loaders(root=root)
    images, labels = next(iter(train_loader))
    plot_samples(images, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = MNISTModel().to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_classifier(model, train_loader, optimizer, nn.CrossEntropyLoss(), epochs=epochs)
    accuracy, errors = evaluate_classifier(model, test_loader, collect_errors=True)
    print(f"Accuracy: {100 * accuracy:.2f}%")
    print(f"Misclassified samples: {len(errors)}")
    plot_misclassified(errors)
    return model


def main():
    examples = {
        "models": show_models,
        "xor": run_xor_comparison,
        "activations": plot_activation_functions,
        "iris": run_iris_classification,
        "tensors": demonstrate_tensors,
        "mnist": run_mnist_classification,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("example", nargs="?", default="all", choices=["all", *examples])
    args = parser.parse_args()
    actions = examples.values() if args.example == "all" else [examples[args.example]]
    for action in actions:
        action()
    plt.show()


if __name__ == "__main__":
    main()
