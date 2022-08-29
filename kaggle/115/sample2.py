from distutils.command.install_egg_info import safe_name
from random import sample
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class AyameDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./train.tsv', skiprows=1, dtype=object)
        train_x = xy[:, 1:5].astype(float)
        train_y = xy[:, 5].reshape(75, 1)
        train_y[train_y == 'Iris-setosa'] = 1
        train_y[train_y == 'Iris-virginica'] = 2
        train_y[train_y == 'Iris-versicolor'] = 3
        train_y = train_y.astype(float)

        self.train_x = torch.tensor(train_x, requires_grad=True, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.sample_size = train_x.shape[0]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.sample_size


class LinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


dataset = AyameDataset()
input_size = 4
hidden_size = 30
output_size = 1
model = LinearRegression(input_size, hidden_size, output_size)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.MSELoss()
epochs = 50000

for epoch in range(epochs):
    inputs, targets = dataset[:]
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    if (epoch + 1) % 1000 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()))


with torch.no_grad():
    test_data = np.loadtxt('./test.tsv', skiprows=1, dtype=object)
    test_x = torch.tensor(test_data[:, 1:5].astype(float), dtype=torch.float32)
    ids = test_data[:, 0].reshape(75, 1).astype(int).astype(object) # -> ndarray

    outputs = torch.round(model(test_x)).numpy().astype(int).astype(object)
    outputs[outputs == 1] = 'Iris-setosa'
    outputs[outputs == 2] = 'Iris-virginica'
    outputs[outputs == 3] = 'Iris-versicolor'

    ans_data = np.concatenate([ids, outputs], axis=1)
    np.savetxt('./ans.csv', ans_data, delimiter=',', fmt="%s")