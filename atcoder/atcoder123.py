a, b, c, d = [int(_) for _ in input().split()]
if a+b == c+d:
    print('Balanced')
if a+b < c+d:
    print('Right')
if a+b > c+d:
    print('Left')