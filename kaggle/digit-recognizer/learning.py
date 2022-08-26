# 疲れたので一旦放置

# 新しくやったこと
# - グラフを重ねて複数表示
# - バッチを導入

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np


xy = np.loadtxt('train.csv', skiprows=1, delimiter=',')
train_size = 33600
test_size  = 8400


class DigitDataset(Dataset):
    def __init__(self):
        train_x = xy[train_size:, 1:]
        train_y = xy[train_size:, 0].reshape(-1, 1)

        self.train_x = torch.tensor(train_x, dtype=torch.float32)
        self.train_y = torch.tensor(train_y, dtype=torch.float32)
        self.train_size = self.train_x.shape[0]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.train_size


class DigitTestDataset(Dataset):
    def __init__(self):
        np.random.shuffle(xy)
        test_x = xy[:train_size, 1:]
        test_y = xy[:train_size, 0].reshape(-1, 1)

        self.test_x = torch.tensor(test_x, dtype=torch.float32)
        self.test_y = torch.tensor(test_y, dtype=torch.float32)
        self.test_size = self.test_x.shape[0]

    def __getitem__(self, index):
        return self.test_x[index], self.test_y[index]

    def __len__(self):
        return self.test_size


class LinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.sigmoid(x)
        x = self.linear2(x)
        x = F.softmax(input=x, dim=1)
        return x


# parameters
input_size  = 28*28
hidden_size = 128
output_size = 10   # -> ここって1でいいの？0 ~ 9の10では？ソフトマックス使うのなら。
batch_size  = 410
lr          = 0.01
epochs      = 500

# for making graphs
epoch_x    = np.array([])
loss_y     = np.array([])
accuracy_y = np.array([])

train_dataset = DigitDataset()
dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

model = LinearRegression(input_size, hidden_size, output_size)
optimizer = torch.optim.SGD(model.parameters(), lr)
critrion = nn.MSELoss()


def calc_accuracy(test_x, test_y):
    test_outputs = torch.round(model(test_x))
    test_size = test_x.size()[0]
    accuracy = float(sum(test_outputs == test_y) / test_size)
    accuracy = round(accuracy, 3)
    return accuracy


for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = critrion(targets, outputs)
        loss.backward()

        optimizer.step()

    if (epoch + 1) % 5 == 0 or epoch == 1:
        test_dataset = DigitTestDataset()
        test_x, test_y = test_dataset[:]
        breakpoint()
        accuracy = calc_accuracy(test_x, test_y)
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()), f'accuracy: {accuracy}')
        epoch_x    = np.append(epoch_x,    epoch)
        loss_y     = np.append(loss_y,     loss.detach().numpy())
        accuracy_y = np.append(accuracy_y, accuracy)


with torch.no_grad():
    epoch_x    = epoch_x.reshape(-1, 1)
    loss_y     = loss_y.reshape(-1, 1)
    accuracy_y = accuracy_y.reshape(-1, 1)

    fig, ax = plt.subplots()
    x = epoch_x
    c1,c2 = "red", "blue"   # 各プロットの色
    l1,l2 = "loss","accuracy"  # 各ラベル

    ax.set_xlabel('epoch')  # x軸ラベル
    ax.set_ylabel('y')      # y軸ラベル
    ax.set_xlim([0, epochs])
    ax.set_ylim([0, 1])

    ax.plot(x, loss_y,     color=c1, label=l1)
    ax.plot(x, accuracy_y, color=c2, label=l2)
    ax.legend(loc=0)
    plt.show()
        
    

        