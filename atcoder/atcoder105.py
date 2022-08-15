n = int(input())
k = int(input())
i = 1

for _ in range(n):
    i = min(i*2, i+k)

print(i)