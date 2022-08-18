n = int(input())
a_list = [int(_) for _ in input().split()]
a_set  = set(a_list)
i_list = []

for s in a_set:
    i = 0
    if a_list.count(s) < s:
        i_list.append(a_list.count(s))
        a_list = [_ for _ in a_list if _ != s]
 
    else:
        while a_list.count(s) != s:
            a_list.remove(s)
            i += 1
        i_list.append(i)

print(sum(i_list))