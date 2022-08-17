import numpy as np

input()
a = np.array([int(_) for _ in input().split()])
i = 0
while True:
    b = list(filter(lambda x: x % 2 != 0, a))
    if b:
        break
    a = a / 2
    i += 1

print(i)