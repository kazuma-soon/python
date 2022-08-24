n = int(input())
s = [input() for _ in range(n)]
m = int(input())
t = [input() for _ in range(m)]

says = set(s)
i = 0

for say in says:
    if t.count(say) > 0:
        i2 = s.count(say) - t.count(say) if s.count(say) - t.count(say) > 0 else 0
        i = i2 if i2 > i else i
    else:
        i2 = s.count(say)
        i = i2 if i2 > i else i

print(i)
    