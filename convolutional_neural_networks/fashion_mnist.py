# -*- coding: utf-8 -*-

# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms # for data preprocessing
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

# %%
train_dataset = torchvision.datasets.FashionMNIST('./data', download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))

# %%
batch_size = 1000
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# %%
labels_map = {0:'T-Shirt', 1:'Trouser', 2:'Pullover', 3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 7:'Sneaker', 8:'Bag', 9:'Ankle Boot'} # 10 classes
fig = plt.figure(figsize=(8,8))
columns = 4
rows = 5
for i in range(1, columns*rows + 1):
    rand_idx = np.random.randint(len(train_dataset))
    img = train_dataset[rand_idx][0][0,:,:] # extract image data [image index][x][image]
    fig.add_subplot(rows, columns, i)
    plt.title(labels_map[train_dataset[rand_idx][1]]) # train_dataset[rand_idx][1] is class number (0~9)
    plt.axis('off')
    plt.imshow(img, cmap='gray')
plt.show()

# %%
class FashionDNN1(nn.Module):
    def __init__(self):
        super(FashionDNN1, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=256)
        self.drop = nn.Dropout(0.25) # Please see Lecutre 16 or you can comment this line out
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
    def forward(self, input_data):
        out = input_data.view(-1, 784)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# %%
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

# %%
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

model = FashionDNN3()
#summary(model, (1, 28, 28))

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionDNN3().to(device)


# loss and optimizer
from torch.optim import optimizer
import torch
import numpy as np

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# train
n_epoch = 10
for epoch in range(1, n_epoch+1):
    train_losses = []
    train_acc = []
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        train_losses.append(loss.item())
        train_acc.append( (torch.max(pred, 1)[1] == y).sum()/batch_size )
        loss.backward()
        optimizer.step()
    print('epoch: {}, loss:{}, acc:{}'.format(epoch, sum(train_losses)/(idx+1), sum(train_acc)/(idx+1) ))

# %%
test_acc = []
for i, (x, y) in enumerate(test_loader):
    x = x.to(device)
    y = y.to(device)
    test_pred = model(x)
    test_acc.append((torch.max(test_pred, 1)[1] == y).sum()/batch_size)
print('test acc:{}'.format(sum(test_acc)/(len(test_acc) )) )
