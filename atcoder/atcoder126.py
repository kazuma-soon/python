a, b = [int(_) for _ in input().split()]
s = input()
if   s[:a].count('-') != 0:
    print('No')
elif s[a] != '-':
    print('No')
elif s[-b:].count('-') != 0:
    print('No')
else:
    print('Yes')
