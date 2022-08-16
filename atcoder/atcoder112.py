n = int(input())

l = [0] * (n+1)
l[0] = 2
l[1] = 1
for _ in range(2, n+1):
    l[_] = l[_-1] + l[_-2]

print(l[n])
