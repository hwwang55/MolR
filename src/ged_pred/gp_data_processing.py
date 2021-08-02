import os
import random
import dgl
import torch
import pickle
import pysmiles
import itertools
import multiprocessing as mp
from data_processing import networkx_to_dgl
from networkx.algorithms.similarity import graph_edit_distance

random.seed(0)


class GEDPredDataset(dgl.data.DGLDataset):
    def __init__(self, args):
        self.args = args
        self.path = '../data/' + args.dataset + '/'
        self.graphs1 = []
        self.graphs2 = []
        self.targets = []
        super().__init__(name='ged_pred_' + args.dataset)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.args.dataset + ' dataset to GPU')
            self.graphs1 = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs1]
            self.graphs2 = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.graphs2]

    def save(self):
        print('saving ' + self.args.dataset + ' dataset to ' + self.path + 'ged0.bin and ' + self.path + 'ged1.bin')
        dgl.save_graphs(self.path + 'ged0.bin', self.graphs1, {'target': self.targets})
        dgl.save_graphs(self.path + 'ged1.bin', self.graphs2)

    def load(self):
        print('loading ' + self.args.dataset + ' dataset from ' + self.path + 'ged0.bin and ' + self.path + 'ged1.bin')
        self.graphs1, self.targets = dgl.load_graphs(self.path + 'ged0.bin')
        self.graphs2, _ = dgl.load_graphs(self.path + 'ged1.bin')
        self.targets = self.targets['target']
        self.to_gpu()

    def process(self):
        print('loading feature encoder from ../saved/' + self.args.pretrained_model + '/feature_enc.pkl')
        with open('../saved/' + self.args.pretrained_model + '/feature_enc.pkl', 'rb') as f:
            feature_encoder = pickle.load(f)

        molecule_list = self.get_molecule_list()
        samples = self.sample(molecule_list)
        res = calculate_ged_with_mp(samples, self.args.n_pairs)

        with open(self.path + 'pairwise_ged.csv', 'w') as f:
            f.writelines('smiles1,smiles2,ged\n')
            for g1, g2, s1, s2, ged in res:
                self.graphs1.append(networkx_to_dgl(g1, feature_encoder))
                self.graphs2.append(networkx_to_dgl(g2, feature_encoder))
                self.targets.append(ged)
                f.writelines(s1 + ',' + s2 + ',' + str(ged) + '\n')
        self.targets = torch.Tensor(self.targets)
        self.to_gpu()

    def has_cache(self):
        if os.path.exists(self.path + 'ged0.bin') and os.path.exists(self.path + 'ged1.bin'):
            print('cache found')
            return True
        else:
            print('cache not found')
            return False

    def __getitem__(self, i):
        return self.graphs1[i], self.graphs2[i], self.targets[i]

    def __len__(self):
        return len(self.graphs1)

    def get_molecule_list(self):
        print('retrieving the first %d molecules from %s dataset' % (self.args.n_molecules, self.args.dataset))
        molecule_list = []
        with open(self.path + self.args.dataset + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0 or line == '\n':
                    continue
                if idx > self.args.n_molecules:
                    break
                items = line.strip().split(',')

                if self.args.dataset == 'QM9':
                    smiles = items[1]
                else:
                    raise ValueError('unknown dataset')

                raw_graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
                molecule_list.append((raw_graph, smiles))
        return molecule_list

    def sample(self, molecule_list):
        print('sampling %d pairs' % self.args.n_pairs)
        all_pairs = list(itertools.combinations(molecule_list, 2))
        samples = random.sample(all_pairs, self.args.n_pairs)
        return samples


def calculate_ged_with_mp(samples, n_pairs):
    print('calculating GED using multiprocessing')
    n_cores, pool, range_list = get_params_for_mp(n_pairs)
    res = pool.map(calculate_ged, zip([samples[i[0]: i[1]] for i in range_list], range(n_cores)))
    print('gathering results')
    res = [i for sublist in res for i in sublist]
    return res


def get_params_for_mp(n_pairs):
    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    avg = n_pairs // n_cores
    range_list = []
    start = 0
    for i in range(n_cores):
        num = avg + 1 if i < n_pairs - avg * n_cores else avg
        range_list.append([start, start + num])
        start += num
    return n_cores, pool, range_list


def calculate_ged(inputs):
    def node_match(n1, n2):
        return n1['element'] == n2['element'] and n1['charge'] == n2['charge']

    def edge_match(e1, e2):
        return e1['order'] == e2['order']

    res = []
    samples, pid = inputs
    for i, graph_pair in enumerate(samples):
        g1, g2 = graph_pair
        graph1, smiles1 = g1
        graph2, smiles2 = g2
        ged = graph_edit_distance(graph1, graph2, node_match=node_match, edge_match=edge_match)
        res.append((graph1, graph2, smiles1, smiles2, ged))
        if i % 100 == 0:
            print('pid %d:  %d / %d' % (pid, i, len(samples)))
    print('pid %d  done' % pid)
    return res


def load_data(args):
    data = GEDPredDataset(args)
    return data
