#%%

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, learning_curve
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
os.chdir('/Users/kazuma/Documents/menew/my_python/kaggle/titanic')


# Read the data
train_df = pd.read_csv('train.csv')
test_df  = pd.read_csv('test.csv')
print(train_df)

# Ready 'Passengerid' for summission.csv
test_passengerId = test_df['PassengerId']

# 不要なカラムを除去
train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# ラベルエンコード
le = preprocessing.LabelEncoder()

train_df['Sex'] = le.fit_transform(train_df['Sex'])
train_df['Embarked']  = le.fit_transform(train_df['Embarked'])

test_df['Sex'] = le.fit_transform(test_df['Sex'])
test_df['Embarked']  = le.fit_transform(test_df['Embarked'])


# fill missing values # ++20%
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Feature Engineering
train_df['Family_size'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['Family_size']  = test_df['SibSp'] + test_df['Parch'] + 1

# Devide into X & Y and Split data into train & test
train_df_y = train_df['Survived']

train_df.drop('Survived', axis=1, inplace=True)
X_train, X_val, Y_train, Y_val = train_test_split(train_df, train_df_y, test_size=0.2, random_state=0)


# Define xgboost model
model = xgb.XGBClassifier(
    max_depth=5,
    n_estimators=100,
    learning_rate=0.01,
    random_state=0
)

# Train model
model.fit(X_train, Y_train)

# Plot importance of features
fig, ax = plt.subplots()
xgb.plot_importance(model, ax=ax)
plt.show()

# Plot accuracy
# Train model and get learning curve
train_sizes, train_scores, test_scores = learning_curve(
    model, X_train, Y_train, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), verbose=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training examples')
plt.ylabel('Score')
plt.legend()
plt.show()


# Evaluate model with val_data
y_pred = model.predict(X_val)
accuracy = accuracy_score(Y_val, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Predict test_df
test_predictions  = model.predict(test_df)

# Create submission.csv
submission_df = pd.DataFrame({"PassengerId": test_passengerId, "Survived": test_predictions})
submission_df.to_csv('submission.csv', index=False)

# %%
