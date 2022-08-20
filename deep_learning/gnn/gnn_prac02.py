import numpy as np

n = 6
E = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (4, 5)]
A = np.zeros((n, n))
for (i, j) in E:
    # 隣接行列は斜めに対象
    A[i, j] = A[j, i] = 1
print(f"{A = }")

X = np.array([
    [0, 0, 0], # 0
    [0, 0, 1], # 1
    [0, 1, 0], # 2
    [0, 1, 1], # 3
    [1, 0, 0], # 4
    [1, 1, 0], # 5
])
d = X.shape[1]
out_dim = 2

W = np.ones((d, out_dim))

Ahat = A + np.eye(n)
Dhat = lambda x: np.diag(np.power(np.sum(Ahat, axis=1), x))
D = np.diag(np.power(np.sum(Ahat, axis=1), -1))

mat0 = Ahat @ X @ W # none
mat2 = (Dhat(-1/2)) @ Ahat @ (Dhat(-1/2)) @ X @ W # both
print(mat0)
print(mat2)