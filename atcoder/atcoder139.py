import numpy as np
n, m = [int(_) for _ in input().split()]
nm   = np.ones((n, m))

def get(lst, index1, index2):
    return lst[index1, index2] if len(lst[0]) > index1 > 0 and len(lst) > index2 > 0 else None

for i in range(n):
    for j in range(m):
        if get(nm, i-1, j-1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
            
        if get(nm, i, j-1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1

        if get(nm, i+1, j-1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
            
        if get(nm, i-1, j) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
            
        if get(nm, i, j) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1

        if get(nm, i+1, j) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
        
        if get(nm, i-1, j+1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
        
        if get(nm, i, j+1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1
        
        if get(nm, i+1, j+1) == 1:
            nm[i][j] = 0
        else:
            nm[i][j] = 1

print(nm.flatten())
