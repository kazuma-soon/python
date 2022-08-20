import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


N = 100
x = np.linspace(-15, 15, N)
y = -2 * x**2 + 32*np.random.randn(N)

x = x.astype('float32')
y = y.astype('float32')

x = torch.from_numpy(x)
y = torch.from_numpy(y)

w0 = torch.tensor(1.0, requires_grad=True)
w1 = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)


def model(x):
  return w0*x**2 + w1*x + b


criterion = nn.MSELoss()
optimizer = torch.optim.SGD([w0, w1, b], lr=1.0e-5)


losses = []
epochs = 10000
es_count = 0
patience = 3

for epoch in range(epochs):
  optimizer.zero_grad()

  pred = model(x.view(-1, 1))

  loss = criterion(pred, y.view(-1, 1))
  loss.backward()
  optimizer.step()

  losses.append(loss.item())
  if epoch % 100 == 0:
    print("epoch {}, loss: {}".format(epoch+1, losses[epoch]))

  # Early Stopping
  if epoch > 0 and losses[epoch - 1] < losses[epoch]:
    es_count += 1
    if es_count >= patience:
      break
  else:
    es_count = 0

print('loss:', loss.item())
print('w0:', w0.item())
print('w1:', w1.item())
print('b:', b.item())
