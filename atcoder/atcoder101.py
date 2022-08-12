n = int(input())
a = [int(x) for x in input().split()]
total = sum(a)
x = 0
ans = 10**10
for i in range(n-1):
    x += a[i]
    ans = min(ans, abs(total - 2*x))

print(ans)


# n=int(input())
# L=list(map(int,input().split()))
# SUM=sum(L)
# ans=10**10
# wa=0
# for i in range(n-1):
#   wa+=L[i]
#   ans=min(ans,abs(SUM-2*wa))
# print(ans)