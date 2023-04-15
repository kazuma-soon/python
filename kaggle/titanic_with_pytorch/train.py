import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('/Users/kazuma/Documents/menew/my_python/kaggle/titanic_with_pytorch')


# Read data
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')

# Get necessary columns before preprocessing
train_df_Survived = train_df['Survived']
test_df_PassengerId = test_df['PassengerId']

# Preprocessing: Delete unnesessary columns
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df  = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Preprocessing: Imputation
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)
test_df['Embarked'].fillna(test_df['Embarked'].mode()[0], inplace=True)

test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace=True)

# Preprocessing: Labeling
le = LabelEncoder()
train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked'] = le.fit_transform(train_df['Embarked'])

test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['Embarked'] = le.fit_transform(test_df['Embarked'])

# Split data into Train & Val
X_train, X_val, Y_train, Y_val = train_test_split(train_df.drop('Survived', axis=1), train_df['Survived'], test_size=0.2, random_state=0)


# Define DatasetClass
class TitanicDataset(Dataset):
    def __init__(self, X, Y=None):
        self.X: pd.DataFrame = X
        self.Y: pd.Series    = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.Y is not None:
            return torch.tensor(self.X.iloc[idx].values), torch.tensor(self.Y.iloc[idx])
        else:
            return torch.tensor(self.X.iloc[idx].values)

# Create Dataset about train, val, test
train_dataset = TitanicDataset(X_train, Y_train)
val_dataset   = TitanicDataset(X_val, Y_val)
test_dataset  = TitanicDataset(test_df)

# Define DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=10, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

# Define model class
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Generate model
input_size = len(X_train.columns)
output_sise = len(Y_train.unique())
model = NeuralNet(input_size=input_size, hidden_size=160, output_size=output_sise)

# Define LossFunction and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.01)

# Train model
train_losses = []
val_lossies  = []
accuracies   = []

for epoch in range(50):
    train_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch.float()).requires_grad_(True)
        loss = criterion(y_pred, Y_batch.long()).requires_grad_(True)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss = train_loss / len(train_loader)


    print(f'epoch: {epoch+1}, train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}')