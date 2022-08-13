from inspect import Parameter
import torch
import torch.nn as nn

x = torch.linspace(-15, 15, 100)
y = 5 * x **2

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w * x**2

print(f'Prediction before training: f(5) = {forward(5).item():.3f}')

# loss & optim
learning_rate= 1.0e-5
n_iters = 10000

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=1.0e-5)

# loop
for epoch in range(n_iters):
    y_pred = forward(x)

    l = loss(y, y_pred)
    l.backward()

    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print('epoch ', epoch+1, ': w = ', w, ' loss = ', l)

print(f'Prediction after training: f(5) = {forward(5).item():.3f}')

