import dgl
import torch

def build_sample_graph():
    edge_list = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3)]
    g = dgl.graph(edge_list)

    src, dst = tuple(zip(*edge_list))
    g.add_edges(dst, src)

    return g


G = build_sample_graph()
G.ndata['h'] = torch.eye(5)

breakpoint()

def gcn_message(edges):
    return {'msg' : edges.src['h']}

G.send(G.edges(), gcn_message)