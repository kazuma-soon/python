n = input()
i = 0
for _ in range(len(n)-1):
    if n[_] == n[_+1]:
        i += 1
    else:
        i = 0
    if i >= 2:
        break

print('Yes' if i >= 2 else 'No')