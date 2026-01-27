# -*- coding: utf-8 -*-

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# Prepare the data
X = torch.tensor([[1200.0, 3], [1500.0, 4], [1800.0, 3], [2000.0, 5]])  # House size (square feet), number of rooms
y = torch.tensor([300000.0, 400000.0, 450000.0, 500000.0])  # House prices

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # Linear layer with 2 input features and 1 output

    def forward(self, x):
        return self.linear(x)  # Forward pass

model = LinearRegressionModel()

# Loss function and optimization algorithm
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.00000001)  # Stochastic Gradient Descent optimizer

# Training loop
for epoch in range(1000):
    optimizer.zero_grad()  # Zero the gradients
    outputs = model(X)  # Forward pass
    loss = criterion(outputs, y.view(-1, 1))  # Compute loss
    loss.backward()  # Backward pass (compute gradients)
    optimizer.step()  # Update parameters
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}')  # Print loss every 100 epochs

# Analyze the results
predicted = model(X).detach().numpy()  # Get the model predictions
print(predicted)  # Print the predicted prices

# %%
from sklearn.model_selection import train_test_split

# Prepare the data
X = [[1200.0, 3], [1500.0, 4], [1800.0, 3], [2000.0, 5], [2200.0, 4], [2400.0, 5], [2600.0, 4], [2800.0, 5], [2400.0, 5], [2600.0, 4]]
y = [300000.0, 400000.0, 450000.0, 500000.0, 550000.0, 600000.0, 650000.0, 700000.0, 600000.0, 650000.0]

# Split the data into training set and temporary set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the temporary set into validation set and test set
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

print(f'Training set size: {len(X_train)}')
print(f'Validation set size: {len(X_val)}')
print(f'Test set size: {len(X_test)}')

# %%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load the data
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Train the logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print performance metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# %%
import numpy as np
import pandas as pd

# Create hypothetical data with random noise
np.random.seed(6370)  # For reproducibility
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# Add some random noise to the scores
scores = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 95]) + np.random.normal(0, 5, len(study_hours))

# Create a DataFrame from the data
data = {
    'study_hours': study_hours,
    'scores': scores
}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# %%
X = df[['study_hours']].values  # Features
y = df['scores'].values  # Labels

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# Data preparation
X_train = torch.tensor(X, dtype=torch.float32)  # Convert input features to PyTorch tensor
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Convert labels to PyTorch tensor and reshape

# Model definition
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Define a linear layer with 1 input and 1 output feature

    def forward(self, x):
        return self.linear(x)  # Define the forward pass

model = LinearRegressionModel()

# Loss function and optimization algorithm
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent optimizer with learning rate of 0.01

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)  # Calculate loss

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    # Print loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Perform predictions
predicted = model(X_train).detach().numpy()  # Detach the predictions from the computational graph and convert to numpy array
print(predicted)

# %%
import matplotlib.pyplot as plt

# Compare predicted values with actual values
plt.plot(df['study_hours'], y, 'ro', label='Actual')  # Plot actual values as red dots
plt.plot(df['study_hours'], predicted, 'b-', label='Predicted')  # Plot predicted values as a blue line
plt.xlabel('Study Hours')  # Label for the x-axis
plt.ylabel('Scores')  # Label for the y-axis
plt.legend()  # Display the legend
plt.show()  # Show the plot
