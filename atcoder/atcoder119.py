import collections

n, k = [int(_) for _ in input().split()]
A = [int(_) for _ in input().split()]
setA = set(A)
groupA = sorted(collections.Counter(A).items(), key=lambda x:x[1])
over = len(setA) - k

if over <= 0:
    print(0)
else:
    i = 0
    for _ in range(over):
        i += groupA[_][1]
    print(i)