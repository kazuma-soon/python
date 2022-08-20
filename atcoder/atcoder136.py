import collections
n = int(input())
names = []
for _ in range(n):
    names.append(input())

march_names = [_ for _ in names if _[0] in 'MARCH']
march_first_string = collections.Counter([_[0] for _ in march_names])
if len(march_first_string) == 0:
    print(0)
else:
    i = 1
    for key, value in march_first_string.items():
        i *= value
    print(i)