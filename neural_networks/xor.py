# -*- coding: utf-8 -*-

# %%
import torch
import torch.nn as nn

from torch.optim.adam import Adam

model = nn.Sequential(
    nn.Linear(2, 2),
    nn.Sigmoid(),
    nn.Linear(2, 1),
    nn.Sigmoid()
)

x = torch.FloatTensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.Tensor([[0.], [1.], [1.], [0.]])

optim = Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optim.zero_grad()
    preds = model(x)
    loss = nn.BCELoss()(preds, y)
    loss.backward()
    optim.step()
    if epoch % 100 == 0:
        print('loss:', loss.item())
# test
out = model(x)
print(out)
