input()
a = sorted([int(_) for _ in input().split()])[::-1]
print(sum(a[::2]) - sum(a[1::2]))
