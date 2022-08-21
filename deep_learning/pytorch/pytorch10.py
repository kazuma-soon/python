from dataclasses import dataclass
from turtle import forward
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# データを標準化する -> ないとlossが減らない
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore


# データの設定
class WineDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/wine.csv', delimiter=',', skiprows=1)
        # シャッフルする -> ないとラベル表現がトレイン・テストで偏る
        np.random.shuffle(xy)
        self.n_sample = xy.shape[0]
        x_data = zscore(xy[:138, 1:])
        y_data = zscore(xy[:138, 0])
        x_test = zscore(xy[139:, 1:])
        y_test = zscore(xy[139:, 0])
        self.x_data = torch.from_numpy(x_data).float()
        self.y_data = torch.from_numpy(y_data).float()
        self.x_test = torch.from_numpy(x_test).float()
        self.y_test = torch.from_numpy(y_test).float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample

    def get_test_data(self):
        return self.x_test, self.y_test


# モデルの定義
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# data x modelを実装
dataset = WineDataset()
input_size = len(dataset[0][0]) # -> 13
output_size= 1
model = LinearRegression(input_size, output_size)

# opt, lossの定義
optimizer = torch.optim.Adam(params = model.parameters(), lr = 0.1)
criterion = nn.MSELoss()


# 学習ループ
epochs = 100000
for epoch in range(epochs):
    inputs, targets = dataset[:]
    targets = targets.reshape(-1, 1)
    inputs.requires_grad_(True)
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()))


x_test, y_test = dataset.get_test_data()
y_test = y_test.reshape(-1, 1)
y_test = y_test.round()
y_test[y_test < 0] = 0
with torch.no_grad():
    y_predicted = model(x_test)
    # ans -> 1, 2, 3なのに、y_predを0 or 1にしてしまっている。。。
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')





# lossが減っていかない
# -> weightの更新を見てみる
# -> optimizer.stepがちゃんと動いているか？
    