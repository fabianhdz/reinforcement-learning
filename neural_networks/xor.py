"""Train a small sigmoid network on XOR."""

import torch
import torch.nn as nn


def create_model():
    return nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(),
                         nn.Linear(2, 1), nn.Sigmoid())


def train_model(model, *, epochs=1000, learning_rate=0.01):
    x = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
    y = torch.tensor([[0.], [1.], [1.], [0.]])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(x)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print("loss:", loss.item())
    with torch.no_grad():
        return model(x)


def main():
    model = create_model()
    print(train_model(model))

if __name__ == "__main__":
    main()
