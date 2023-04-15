# モデルが十分にデータを再現できない。特徴量は全部必要？

# train自体のaccuracyを出してみて、高すぎるとカガクシュウ -> カガクシュウしている。
# Epoch [149000/200000], Loss: 0.0075 self_accuracy: 1.0 accuracy: 0.8947368264198303

# 特徴量同士の相関高いやつを片方？除く
# Flavanoidsを除いてみる

# 解決
# 学習用データをシャッフルしていなかった。
# そのため、過学習が発生・汎用的な最適値を出せなかった。
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.chdir('/Users/kazuma/Documents/menew/my_python/kaggle/113')


class WineDataset(Dataset):
    def __init__(self, test_mode=None):
        # 学習用：テスト用 = 70：19で分ける
        xy = np.loadtxt('train.tsv', skiprows=1)
        np.random.shuffle(xy)
        # xy = np.delete(arr=xy, obj=[7, 8, 13], axis=1)
        self.train_x = torch.tensor(xy[:70, 2:], dtype=torch.float32)
        self.train_y = torch.tensor(xy[:70, 1], dtype=torch.float32).reshape(70, 1)
        self.test_x = torch.tensor(xy[70:, 2:], dtype=torch.float32)
        self.test_y = torch.tensor(xy[70:, 1], dtype=torch.float32).reshape(19, 1)
        self.sample_size = self.train_x.shape[0]
        self.test_mode = test_mode

    def __getitem__(self, index):
        if self.test_mode:
            return self.test_x[index], self.test_y[index]
        else:
            return self.train_x[index], self.train_y[index]

    def __len__(self):
        return self.sample_size


class LinearRegression(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LinearRegression, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size1)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


dataset = WineDataset()
input_size   = 13
hidden_size1 = 64
hidden_size2 = 32
output_size  = 1
lr           = 0.01
epochs       = 100000

model = LinearRegression(input_size, hidden_size1, hidden_size2, output_size)
optimizer = torch.optim.Adadelta(params=model.parameters(), lr=lr)
criterion = nn.MSELoss()

plot_x = np.array([])
plot_y = np.array([])


# 正答率の計算
def calc_accuracy(test_x, test_y):
    test_outputs   = torch.round(model(test_x))
    test_size = test_y.size()[0]
    accuracy = float(sum(test_y == test_outputs) / test_size)
    return accuracy


# 過学習を検知するため、学習したデータ自体への正答率を計算する
def calc_self_accuracy(train_x, train_y, outputs):
    outputs = torch.round(outputs).to(torch.int)
    train_size = train_y.size()[0]
    self_accuracy = float(sum(train_y == outputs) / train_size)
    return self_accuracy


for epoch in range(epochs):
    inputs, targets = dataset[:]
    optimizer.zero_grad()
    outputs = model(inputs)

    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()

    # lossの推移をプロットするため、x軸をエポックのリスト, y軸をlossのリストとして作成する
    plot_x = np.append(plot_x, epoch)
    plot_y = np.append(plot_y, loss.detach().numpy())

    if (epoch + 1) % 500 == 0 or epoch == 1:
        self_accuracy = calc_self_accuracy(inputs, targets, outputs)

        test_dataset = WineDataset(test_mode=True)
        test_x, test_y = test_dataset[:]
        accuracy = calc_accuracy(test_x, test_y)

        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()), f'self_accuracy: {self_accuracy}', f'accuracy: {accuracy}')
        


with torch.no_grad():
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(0.0, 1.0)
    plt.plot(plot_x, plot_y)
    plt.show()


# 回答作成のため、ans.csvにid, classを書き込む
with torch.no_grad():
    test_data = np.loadtxt('test.tsv', skiprows=1)
    ids       = torch.tensor(test_data[:, 0], dtype=torch.int).reshape(89, 1)
    inputs  = torch.tensor(test_data[:, 1:], dtype=torch.float32)
    outputs = model(inputs)
    outputs = torch.round(outputs).to(torch.int).numpy()

    ans_data = np.concatenate([ids, outputs], axis=1)
    np.savetxt('ans.csv', ans_data,  delimiter=',', fmt='%s')




