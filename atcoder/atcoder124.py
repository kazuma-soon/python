n, a, b = [int(_) for _ in input().split()]
total = 0
def sum_digits(n):
    s = [int(_) for _ in str(n)]
    return sum(s)

for i in range(1, n+1):
    digits = sum_digits(i)
    if a <= digits <= b:
        total += i

print(total)

