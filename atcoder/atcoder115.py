n = input()
ds = [int(_) for _ in n]
n = int(n)
ds_sum = sum(ds)

print("Yes" if n % ds_sum == 0 else "No")

