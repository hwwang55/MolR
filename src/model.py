import torch
from dgl.nn import GraphConv, GATConv
from dgl.nn.pytorch.glob import SumPooling
from torch.nn import Embedding, ModuleList


class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layer, n_values, emb_dim, dist_metric):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.dim = emb_dim
        self.dist_metric = dist_metric
        self.embed_layer = Embedding(n_values, emb_dim)
        self.gnn_layers = ModuleList([])
        for i in range(n_layer):
            if gnn == 'gcn':
                if i == 0:
                    self.gnn_layers.append(GraphConv(emb_dim, emb_dim, weight=False, bias=False))
                else:
                    self.gnn_layers.append(GraphConv(emb_dim, emb_dim, bias=False))
            elif gnn == 'gat':
                self.gnn_layers.append(GATConv(emb_dim, emb_dim, num_heads=1))
            else:
                raise ValueError('unknown GNN model')
        self.pooling_layer = SumPooling()

    def forward(self, graph):
        feature = graph.ndata['feature']
        h = self.embed_layer(feature)
        h = torch.sum(h, dim=1)
        for i in range(self.n_layer - 1):
            h = self.gnn_layers[i](graph, h)
            h = torch.relu(h)
        h = self.gnn_layers[-1](graph, h)
        graph_embedding = self.pooling_layer(graph, h)

        if self.gnn == 'gat':
            graph_embedding = torch.squeeze(graph_embedding)
        return graph_embedding
