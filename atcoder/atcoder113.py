num = input()
a, b, c, d = [int(_) for _ in num]

ops = [
    ["+", "+", "+"],
    ["+", "+", "-"],
    ["+", "-", "+"],
    ["+", "-", "-"],
    ["-", "+", "+"],
    ["-", "+", "-"],
    ["-", "-", "+"],
    ["-", "-", "-"],

]
for op in ops:
    s = f"{a}{op[0]}{b}{op[1]}{c}{op[2]}{d}"
    i = eval(s)
    if i == 7:
        s = s + '=7'
        break

print(s)
    
    