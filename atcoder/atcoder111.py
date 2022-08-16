n = int(input())
l = [2, 1]
if n >= 2:
    for j in range(2, n+1):
        i = l[j-1] + l[j-2]
        l.append(i)
if n == 1:
    i = 1

print(i)