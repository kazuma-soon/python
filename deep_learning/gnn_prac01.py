# https://zenn.dev/takilog/articles/e54a45d6f7266229e367

import numpy as np

n = 6
output_dim = 2
E = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (4, 5)]
A = np.zeros((n, n))
Ahat = A + np.eye(n)

# np.sum(Ahat, axis=1) = array([1., 1., 1., 1., 1., 1.])
Dhat = lambda x: np.diag(np.power(np.sum(Ahat, axis=1), x))

# 入力X
X = np.array([
    [0, 0, 0], # 0
    [0, 0, 1], # 1
    [0, 1, 0], # 2
    [0, 1, 1], # 3
    [1, 0, 0], # 4
    [1, 1, 0], # 5
])
d = X.shape[1] # d = 6
W = np.ones((d, output_dim)) # W.shape = (3, 2)
breakpoint()
# 行列計算
# @機能している
Omat = Dhat(-1/2) @ Ahat @ Dhat(-1/2) @ X @ W
print(Omat)

"""出力
[[0.59153222 0.59153222]
 [0.59153222 0.59153222]
 [1.34885331 1.34885331]
 [1.31622777 1.31622777]
 [1.4080288  1.4080288 ]
 [1.40824829 1.40824829]]
"""