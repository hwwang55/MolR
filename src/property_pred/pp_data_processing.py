import os
import dgl
import torch
import pickle
import pysmiles
from data_processing import networkx_to_dgl


class PropertyPredDataset(dgl.data.DGLDataset):
    def __init__(self, args):
        self.args = args
        self.path = '../data/' + args.dataset + '/' + args.dataset
        self.graphs = []
        self.labels = []
        super().__init__(name='property_pred_' + args.dataset)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.args.dataset + ' dataset to GPU')
            self.graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs]

    def save(self):
        print('saving ' + self.args.dataset + ' dataset to ' + self.path + '.bin')
        dgl.save_graphs(self.path + '.bin', self.graphs, {'label': self.labels})

    def load(self):
        print('loading ' + self.args.dataset + ' dataset from ' + self.path + '.bin')
        self.graphs, self.labels = dgl.load_graphs(self.path + '.bin')
        self.labels = self.labels['label']
        self.to_gpu()

    def process(self):
        print('loading feature encoder from ../saved/' + self.args.pretrained_model + '/feature_enc.pkl')
        with open('../saved/' + self.args.pretrained_model + '/feature_enc.pkl', 'rb') as f:
            feature_encoder = pickle.load(f)
        print('processing ' + self.args.dataset + ' dataset')
        with open(self.path + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                items = line.strip().split(',')
                if self.args.dataset == 'BBBP':
                    smiles, label = items[-1], items[-2]
                    # the next line is to remove unnecessary hydrogen atoms that will cause discontinuous node labels
                    smiles = smiles.replace('([H])', '').replace('[H]', '')
                elif self.args.dataset == 'HIV':
                    smiles, label = items[0], items[-1]
                    smiles = smiles.replace('se', 'Se').replace('te', 'Te')
                elif self.args.dataset == 'BACE':
                    smiles, label = items[0], items[2]
                elif self.args.dataset == 'Tox21':
                    smiles, label = items[-1], items[11]
                    smiles = smiles.replace('se', 'Se')
                    if label == '':
                        continue
                elif self.args.dataset == 'ClinTox':
                    smiles, label = items[0], items[2]
                    smiles = smiles.replace('[H]', '')
                else:
                    raise ValueError('unknown dataset')
                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                dgl_graph = networkx_to_dgl(raw_graph, feature_encoder)
                self.graphs.append(dgl_graph)
                self.labels.append(float(label))
        self.labels = torch.Tensor(self.labels)
        self.to_gpu()

    def has_cache(self):
        if os.path.exists(self.path + '.bin'):
            print('cache found')
            return True
        else:
            print('cache not found')
            return False

    def __getitem__(self, i):
        return self.graphs[i], self.labels[i]

    def __len__(self):
        return len(self.graphs)


def load_data(args):
    data = PropertyPredDataset(args)
    return data
