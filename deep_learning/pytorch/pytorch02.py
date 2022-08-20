'''ライブラリの準備'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

'''データセットの準備'''
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

'''データの読み込み'''
wine = load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_class = pd.DataFrame(wine.target, columns=['class'])

'''学習データの準備'''
wine_cat = pd.concat([wine_df, wine_class], axis=1)
wine_cat.drop(wine_cat[wine_cat['class'] == 2].index, inplace=True)
wine_data = wine_cat.values[:,:13]
wine_target = wine_cat.values[:,13]

'''データセットの分割'''
Train_X, Test_X, Train_Y, Test_Y = train_test_split(wine_data, wine_target, test_size=0.25)

'''PyTorch tensorへ変換(型指定)'''
train_X = torch.FloatTensor(Train_X)
train_Y = torch.LongTensor(Train_Y)
test_X = torch.FloatTensor(Test_X)
test_Y = torch.LongTensor(Test_Y)

# tuple: (--train_X--, --train_Y--)
train = TensorDataset(train_X, train_Y)

# ミニバッチ, シャッフルなどを行い「ぶちこむ」準備をする
train_loader = DataLoader(train, batch_size=8, shuffle=True)

'''モデルの定義'''
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # batch_size x 13 x 128 -- 128 x 2 -- -> 8 x 2
    self.fc1 = nn.Linear(13, 128)
    # クラス：0 or 1 の二択。出力は2つ。
    self.fc2 = nn.Linear(128, 2)
  def forward(self, x):
    # ここで、x.shape = (8, 13) 8個ずつデータが計算される。
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
net = Net()

'''最適化手法の定義'''
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

'''学習'''
for epoch in range(100):
  total_loss = 0
  for train_x, train_y in train_loader:
    optimizer.zero_grad()
    loss = criterion( net(train_x), train_y )
    loss.backward()
    optimizer.step()
    total_loss += loss.data
  if (epoch+1) % 10 == 0:
    print(f"{(epoch+1) = }, {loss.item() = }")
    
test_net = net(test_X).detach()
'''精度を計算'''
result = torch.max(test_net, 1)[1] 
accuracy = sum(test_Y.data.numpy() == result.numpy()) / len(test_Y.data.numpy())
print(accuracy)

