# -*- coding: utf-8 -*-

# %%
import torch
import torch.nn as nn

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

model = SimpleNN()
print(model)

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define single-layer perceptron model
class SingleLayerPerceptron(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptron, self).__init__()
        self.fc = nn.Linear(2, 1)  # Single linear layer

    def forward(self, x):
        x = torch.sigmoid(self.fc(x))  # Output layer with Sigmoid activation
        return x

model = SingleLayerPerceptron()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction results
with torch.no_grad():
    predicted = model(X).round()
    print(f'Predicted (Single-Layer):\n{predicted}')

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define MLP model
class XOR_MLP(nn.Module):
    def __init__(self):
        super(XOR_MLP, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # From input layer to hidden layer
        self.fc2 = nn.Linear(2, 1)  # From hidden layer to output layer

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # Hidden layer with Sigmoid activation
        x = torch.sigmoid(self.fc2(x))  # Output layer with Sigmoid activation
        return x

model = XOR_MLP()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Training
num_epochs = 50000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Prediction results
with torch.no_grad():
    predicted = model(X).round()
    print(f'Predicted:\n{predicted}')

# %%
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.linspace(-10, 10, 100)
y = sigmoid(x)

plt.plot(x, y)
plt.title('Sigmoid Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 100)
y = tanh(x)

plt.plot(x, y)
plt.title('Tanh Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

x = np.linspace(-10, 10, 100)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

plt.plot(x, y_relu, label='ReLU')
plt.plot(x, y_leaky_relu, label='Leaky ReLU')
plt.title('ReLU and Leaky ReLU Functions')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid()
plt.show()

# %%
import torch
import torch.nn as nn

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

model = ImageClassifier()
print(model)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Loading the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target  # Now including all three classes: 0, 1, 2

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converting to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # No need to unsqueeze, using CrossEntropyLoss
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)  # No need to unsqueeze, using CrossEntropyLoss

# Creating DataLoader for training data
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Using batch size of 1 for pure SGD

# Defining the model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(4, 3)  # 4 input features, 3 output classes

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.005)

# Training loop
num_epochs = 5000
for epoch in range(num_epochs):
    for batch_x, batch_y in train_dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluating on the test set
with torch.no_grad():
    test_outputs = model(X_test)
    _, test_predictions = torch.max(test_outputs, 1)  # Get the index of the max log-probability
    accuracy = (test_predictions == y_test).sum().item() / y_test.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# %%
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
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

model = SimpleNN()
print(model)

# %%
import torch

# Creating a tensor
x = torch.tensor([1, 2, 3])
print(x)

# Creating a 3x3 matrix filled with zeros
x = torch.zeros(3, 3)
print(x)

# Creating a 2x2 matrix filled with ones
x = torch.ones(2, 2)
print(x)

# Creating a 2x2 matrix with random values from a normal distribution
x = torch.randn(2, 2)
print(x)

# %%
# Tensor addition
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(c)  # tensor([5, 7, 9])

# Tensor multiplication
d = a * b
print(d)  # tensor([ 4, 10, 18])

# %%
# Move tensor to GPU if available
x = torch . tensor ([1 , 2, 3])
if torch.cuda.is_available():
    device = torch.device('cuda')
    x = x.to(device)
    print(x)

# %%
# Example of automatic differentiation
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x + 2
z = y * y * 2
z = z.mean()

# Perform backpropagation
z.backward()
print(x.grad)  # tensor([4.0000, 5.3333, 6.6667])

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Data augmentation and preprocessing settings
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Set data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Visualize some data samples
example_data, example_targets = next(iter(train_loader))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
    plt.title(f'Label: {example_targets[i].item()}')
    plt.axis('off')
plt.show()


# Design MLP model
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


# Initialize model and move to GPU
model = MNISTModel().to(device)
print(model)

# Set loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# Evaluate model performance
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# Analyze misclassified samples
misclassified = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(targets)):
            if predicted[i] != targets[i]:
                misclassified.append((inputs[i], predicted[i], targets[i]))

print(f'Misclassified samples: {len(misclassified)}')

# Visualize misclassified samples
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(misclassified[i][0].cpu().numpy().reshape(28, 28), cmap='gray')
    plt.title(f'Pred: {misclassified[i][1].item()}, True: {misclassified[i][2].item()}')
    plt.axis('off')
plt.show()
