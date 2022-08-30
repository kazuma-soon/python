import dgl
import torch

# edges 0->1, 0->2, 0->3, 1->3
u = torch.tensor([0, 0, 0, 1])
v = torch.tensor([1, 2, 3, 3])
g = dgl.graph((u, v))

print(dir(g))
print()

# graph
print(g)
print()

# nodes
print(g.nodes())
print()


# edgeの表し方
  # 〇〇 から 〇〇 へ -> (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
  # edgeIDを使う     -> tensor([0, 1, 2, 3])

# edge & nodes
# -> (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
print(g.edges())
print()

# edge & nodes & edgeIDs
# -> (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]), tensor([0, 1, 2, 3]))
print(g.edges(form='all'))
print()

# 3番目のedge
# -> tensor(2)
print(g.edges(form='all')[2][2])

 # If the node with the largest ID is isolated (meaning no edges),
# then one needs to explicitly set the number of nodes
g = dgl.graph((u, v), num_nodes=10)

print(g)
