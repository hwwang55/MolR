import pickle
import dgl
import torch
import pysmiles
import numpy as np
from model import GNN
from dgl.dataloading import GraphDataLoader
from data_processing import networkx_to_dgl


class GraphDataset(dgl.data.DGLDataset):
    def __init__(self, path_to_model, smiles_list, gpu):
        self.path = path_to_model
        self.smiles_list = smiles_list
        self.gpu = gpu
        self.parsed = []
        self.graphs = []
        super().__init__(name='graph_dataset')

    def process(self):
        with open(self.path + '/feature_enc.pkl', 'rb') as f:
            feature_encoder = pickle.load(f)
        for i, smiles in enumerate(self.smiles_list):
            try:
                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
                self.graphs.append(dgl_graph)
                self.parsed.append(i)
            except:
                print('ERROR: No. %d smiles is not parsed successfully' % i)
        print('the number of smiles successfully parsed: %d' % len(self.parsed))
        print('the number of smiles failed to be parsed: %d' % (len(self.smiles_list) - len(self.parsed)))
        if torch.cuda.is_available() and self.gpu is not None:
            self.graphs = [graph.to('cuda:' + str(self.gpu)) for graph in self.graphs]

    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


class MolEFeaturizer(object):
    def __init__(self, path_to_model, gpu=0):
        self.path_to_model = path_to_model
        self.gpu = gpu
        with open(path_to_model + '/hparams.pkl', 'rb') as f:
            hparams = pickle.load(f)
        self.mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
        self.dim = hparams['dim']
        if torch.cuda.is_available() and gpu is not None:
            self.mole.load_state_dict(torch.load(path_to_model + '/model.pt'))
            self.mole = self.mole.cuda(gpu)
        else:
            self.mole.load_state_dict(torch.load(path_to_model + '/model.pt', map_location=torch.device('cpu')))

    def transform(self, smiles_list, batch_size=None):
        data = GraphDataset(self.path_to_model, smiles_list, self.gpu)
        dataloader = GraphDataLoader(data, batch_size=batch_size if batch_size is not None else len(smiles_list))
        all_embeddings = np.zeros((len(smiles_list), self.dim), dtype=float)
        flags = np.zeros(len(smiles_list), dtype=bool)
        res = []
        with torch.no_grad():
            self.mole.eval()
            for graphs in dataloader:
                graph_embeddings = self.mole(graphs)
                res.append(graph_embeddings)
            res = torch.cat(res, dim=0).cpu().numpy()
        all_embeddings[data.parsed, :] = res
        flags[data.parsed] = True
        print('done\n')
        return all_embeddings, flags


def example_usage():
    model = MolEFeaturizer(path_to_model='../saved/gcn_1024')
    embeddings, flags = model.transform(['C', 'CC', 'ccc'])
    print(embeddings)
    print(flags)


if __name__ == '__main__':
    example_usage()
