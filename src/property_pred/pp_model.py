import torch
from torch.nn import Linear


class PropertyPredictionModel(torch.nn.Module):
    def __init__(self, mole):
        super(PropertyPredictionModel, self).__init__()
        self.mole = mole
        self.dense = Linear(mole.dim, 1)

    def forward(self, graph):
        graph_embedding = self.mole(graph)
        pred = self.dense(graph_embedding)
        return pred
