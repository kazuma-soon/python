#%%

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import os
os.chdir('/Users/kazuma/Documents/menew/my_python/kaggle/titanic')


# Read the data
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')


# 不要なカラムを除去
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ラベルエンコード
le = preprocessing.LabelEncoder()

train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked']  = le.fit_transform(train_df['Embarked'])

test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['Embarked']  = le.fit_transform(test_df['Embarked'])


# 欠損値を処置 # ++20%
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# 特徴量エンジニアリング

# データを学習用・検証用に分割
train_df_y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)
X_train, X_val, Y_train, Y_val = train_test_split(train_df, train_df_y, test_size=0.2, random_state=0)


# Define xgboost model
model = xgb.XGBClassifier(
    max_depth=3,
    n_estimators=100,
    learning_rate=0.1,
    random_state=0
)

# Grid
hyper_params = {
    'n_estimators': [500, 1000],
    'max_depth': [4, 5],
    'learning_rate': [0.01, 0.05]

}

grid = GridSearchCV(estimator=model, param_grid=hyper_params, cv=5, scoring='accuracy')

# Search best hyper_params and train model
grid.fit(X_train, Y_train)

# Print best hyperparameters
print("Best parameters found: ", grid.best_params_)


# Evaluate model with val_data
model = grid.best_estimator_
y_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# %%
