import numpy as np
import networkx as nx
import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn

n = 6
nxg = nx.Graph()
nxg.add_nodes_from(range(n))
E = [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4), (4, 5)]
for (u, v) in E:
    nxg.add_edge(u, v)
    
g = dgl.from_networkx(nxg)
g = dgl.add_self_loop(g)
print(f"{nxg = }")
print(f"{g = }\n")

X = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], 
                  [0, 1, 1], [1, 0, 0], [1, 1, 0]], dtype=torch.float32)
output_dim = 2

gcn = GraphConv(X.shape[1], output_dim, bias=False)
print(f"{gcn = }\n")
print(f"{gcn.weight = }\n")
print(f"{gcn(g, X) = }\ns")

org_weight = gcn.state_dict()["weight"]
print(f"{list(gcn.parameters()) = }\n")

## another example
# Wparam = torch.nn.parameter.Parameter(
#     torch.tensor([[0., 1.], [2., 3.], [4., 5.]], dtype=torch.float32, requires_grad=True)
# )

Wparam = torch.nn.parameter.Parameter(torch.ones_like(org_weight))
gcn.weight = Wparam
print(f"{gcn.weight=}\n")

print(f"{X = }\n")
print(f"{gcn.weight = }\n")
print(f"{gcn(g, X)}\n")