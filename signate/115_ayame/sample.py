import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

# df_train = pd.read_table('./train.tsv')
# df_train_x = df_train.copy()
# df_train_y = df_train.copy()
# df_train_y[df_train_y['class'] == 'Iris-setosa'] = 1
# df_train_y[df_train_y['class'] == 'Iris-virginica'] = 2
# df_train_y[df_train_y['class'] == 'Iris-versicolor'] = 3
# 
# df_test = pd.read_table('./test.tsv')
# 
# df_sample = pd.read_csv('./sample_submit.csv', names=['id', 'class'])
# df_sample[df_sample.iloc[:, 1] == 'Iris-setosa'] = 1
# df_sample[df_sample.iloc[:, 1] == 'Iris-virginica'] = 2
# df_sample[df_sample.iloc[:, 1] == 'Iris-versicolor'] = 3
# 
# # create data in tensor
# train_x = torch.tensor(df_train_x.iloc[:, 1:5].astype(float).values.tolist())
# train_y = torch.tensor(df_train_y['class'].astype(float)).reshape(75, -1)
# 
# test_x  = torch.tensor(df_test.iloc[:, 1:5].astype(float).values.tolist())
# test_y = torch.tensor(df_sample.iloc[:, 1].astype(float)).reshape(75, -1)

class AyameDataset(Dataset):
    def __init__(self):
        breakpoint()
        df_train = pd.read_table('./train.tsv')
        df_train_x = df_train.copy()
        df_train_y = df_train.copy()
        df_train_y[df_train_y['class'] == 'Iris-setosa'] = 1
        df_train_y[df_train_y['class'] == 'Iris-virginica'] = 2
        df_train_y[df_train_y['class'] == 'Iris-versicolor'] = 3

        df_test = pd.read_table('./test.tsv')

        df_sample = pd.read_csv('./sample_submit.csv', names=['id', 'class'])
        df_sample[df_sample.iloc[:, 1] == 'Iris-setosa'] = 1
        df_sample[df_sample.iloc[:, 1] == 'Iris-virginica'] = 2
        df_sample[df_sample.iloc[:, 1] == 'Iris-versicolor'] = 3        

        self.train_x = torch.tensor(df_train_x.iloc[:, 1:5].astype(float).values.tolist(), dtype=torch.float32).requires_grad_(True)
        self.train_y = torch.tensor(df_train_y['class'].astype(float), dtype=torch.float32).reshape(75, -1)
        self.test_x  = torch.tensor(df_test.iloc[:, 1:5].astype(float).values.tolist(), dtype=torch.float32)
        self.test_y = torch.tensor(df_sample.iloc[:, 1].astype(float), dtype=torch.float32).reshape(75, -1)
        self.n_sample = self.train_x.size()[0]

    def __getitem__(self, index):
        return self.train_x[index], self.train_y[index], self.test_x[index], self.test_y[index]

    def __len__(self):
        return self.n_sample


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
input_size = len(dataset[0][0])
hidden_size = 30
output_size = 1
model = LinearRegression(input_size, hidden_size, output_size)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = nn.MSELoss()

epochs = 50000
for epoch in range(epochs):
    inputs, targets, _, _ = dataset[:]
    outputs = model(inputs)
    optimizer.zero_grad()
    loss = criterion(outputs, targets)
    loss.backward()

    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, epochs, loss.item()))

with torch.no_grad():
    df_test = pd.read_table('./test.tsv')
    ids = torch.tensor(df_test.iloc[:, 0].astype(float).values.tolist(), dtype=torch.float32).reshape(-1, 1).numpy().astype(np.int).astype(np.unicode)
    
    _, _, test_x, _ = dataset[:]
    outputs = torch.round(model(test_x)).numpy().astype(np.int).astype(np.unicode)
    for i, output in enumerate(outputs):
        if output == '1':
            outputs[i] = 'Iris-setosa'
        if output == '2':
            outputs[i] = 'Iris-virginica'
        if output == '3':
            outputs[i] = 'Iris-versicolor'
    
    ans_data = np.concatenate([ids, outputs], axis=1)
    np.savetxt('./ans.csv', ans_data, delimiter=',', fmt="%s")

    

# x_test, y_testの順番がバラバラなため、accuracyが求められない。
# with torch.no_grad():
#     _, _, x_test, y_test = dataset[:]
#     y_pred = torch.round(model(x_test))
#     acc = y_pred.eq(y_test).sum() / len(dataset)
#     print(f'accuracy: {acc.item():.4f}')

