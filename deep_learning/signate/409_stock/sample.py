import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pandas as pd

x = pd.read_csv('./train.csv')

class StockDataset(Dataset):

    def __init__(self):
        self.x_data = pd.read_csv('./train.csv', usecols=['Date', 'Open', 'High', 'Low', 'Close'])
        self.y_data = pd.read_csv('./train.csv', usecols=['Up'])

    def __getitem__(self, index):
        return list(self.x_data.iloc[index]), list(self.y_data.iloc[index])

    def __len__(self):
        return x.shape[0]


dataset = StockDataset()
print(dataset[1])
breakpoint()
