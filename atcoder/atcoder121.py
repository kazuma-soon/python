s = sorted(input())
t = sorted(input(), reverse=True)
s = ''.join(s)
t = ''.join(t)
print('Yes' if s < t else 'No')
