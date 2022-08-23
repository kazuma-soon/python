# ミニバッチの何がメリットなのか？SGD, Adamなどの使い分けは？
# 正答率を上げる
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('train.tsv', skiprows=1)
        self.train_x = torch.tensor(xy[:, 2:], dtype=torch.float32, requires_grad=True)
        self.train_y = torch.tensor(xy[:, 1], dtype=torch.float32).reshape(89, 1)
        self.sample_size = self.train_x.shape[0]

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


dataset = WineDataset()
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
input_size  = 13
hidden_size = 64 # ここのサイズを変えても、lossが微妙な位置で収束するのを妨げれない
output_size = 1
model = LinearRegression(input_size, hidden_size, output_size)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 50000
for epoch in range(epochs):
    for data in dataloader:
        inputs, targets = data[:]
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

    if (epoch + 1) % 500 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()))

with torch.no_grad():
    test_data = np.loadtxt('test.tsv', skiprows=1)
    ids       = torch.tensor(test_data[:, 0], dtype=torch.int).reshape(89, 1)
    inputs  = torch.tensor(test_data[:, 1:], dtype=torch.float32)
    outputs = model(inputs)
    outputs = torch.round(outputs).to(torch.int).numpy()

    ans_data = np.concatenate([ids, outputs], axis=1)
    np.savetxt('ans.csv', ans_data,  delimiter=',', fmt='%s')




