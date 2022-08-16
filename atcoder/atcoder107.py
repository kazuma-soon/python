n = int(input())

ans = 0
for i in range(1, n+1):
    ii = i**2
    if ii <= n:
        ans = ii
    else:
        break

print(ans)