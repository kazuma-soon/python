# xgboostに必要なライブラリをインポート
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import os


# データの読み込み
train_df = pd.read_csv('train.csv')

# データの前処理
train_df = train_df.drop(['Id'], axis=1)
train_df = train_df.dropna()
train_df = pd.get_dummies(train_df)


