import os
import dgl
import torch
import pysmiles
from collections import defaultdict


attribute_names = ['element', 'charge', 'aromatic', 'hcount']


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, feature_encoder, raw_graphs):
        self.args = args
        self.mode = mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.path = '../data/' + self.args.dataset + '/cache/' + self.mode
        self.reactant_graphs = []
        self.product_graphs = []
        super().__init__(name='Smiles_' + mode)

    def to_dgl_graph(self, raw_graph):
        # add edges
        src = [s for (s, _) in raw_graph.edges]
        dst = [t for (_, t) in raw_graph.edges]
        graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
        # add node features
        node_features = []
        for i in range(len(raw_graph.nodes)):
            raw_feature = raw_graph.nodes[i]
            numerical_feature = [self.feature_encoder[j][raw_feature[j]] for j in attribute_names]
            node_features.append(numerical_feature)
        node_features = torch.tensor(node_features)
        graph.ndata['feature'] = node_features
        # transform to bi-directed graph with self-loops
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph = dgl.add_self_loop(graph)
        return graph

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU...')
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.product_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graphs]

    def save(self):
        print('saving ' + self.mode + ' data to disk...')
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.product_graphs)

    def load(self):
        print('loading ' + self.mode + ' data from disk...')
        # graphs loaded from disk will have a default empty label set: [graphs, labels], so we only take the first item
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.product_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.to_gpu()

    def process(self):
        print('processing ' + self.mode + ' data...')
        for i, (raw_reactant_graph, raw_product_graph) in enumerate(self.raw_graphs):
            if i % 10000 == 0:
                print('%dk' % (i // 1000))
            # transform networkx graphs to dgl graphs
            reactant_graph = self.to_dgl_graph(raw_reactant_graph)
            product_graph = self.to_dgl_graph(raw_product_graph)
            self.reactant_graphs.append(reactant_graph)
            self.product_graphs.append(product_graph)
        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.product_graphs[i]

    def __len__(self):
        return len(self.reactant_graphs)


def read_data(dataset, mode, all_values):
    print('preprocessing %s data...' % mode)

    # lists for saving all possible values of each attribute
    graphs = []
    with open('../data/' + dataset + '/' + mode + '.csv') as f:
        for line in f.readlines():
            idx, product_smiles, reactant_smiles, _ = line.strip().split(',')

            # skip the first line
            if len(idx) == 0:
                continue

            if int(idx) % 10000 == 0:
                print('%dk' % (int(idx) // 1000))

            # pysmiles.read_smiles() will raise a ValueError: "The atom [se] is malformatted" on USPTO-479k dataset.
            # This is because "Se" is in a aromatic ring, so in USPTO-479k, "Se" is transformed to "se" to satisfy
            # SMILES rules. But pysmiles does not treat "se" as a valid atom and raise a ValueError. To handle this
            # case, I transform all "se" to "Se" in USPTO-479k.
            if '[se]' in reactant_smiles:
                reactant_smiles = reactant_smiles.replace('[se]', '[Se]')
            if '[se]' in product_smiles:
                product_smiles = product_smiles.replace('[se]', '[Se]')

            # use pysmiles.read_smiles() to parse SMILES and get graph objects (in networkx format)
            reactant_graph = pysmiles.read_smiles(reactant_smiles, zero_order_bonds=False)
            product_graph = pysmiles.read_smiles(product_smiles, zero_order_bonds=False)

            # store all values
            for graph in [reactant_graph, product_graph]:
                for attr in attribute_names:
                    for _, value in graph.nodes(data=attr):
                        all_values[attr].add(value)

            graphs.append([reactant_graph, product_graph])

    return all_values, graphs


def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    # key: attribute; values: all possible values of the attribute
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1

    return feature_encoder, idx


def preprocess(dataset):
    print('preprocessing %s dataset...' % dataset)

    # read all data and get all values for attributes
    all_values = defaultdict(set)
    all_values, train_graphs = read_data(dataset, 'train', all_values)
    all_values, valid_graphs = read_data(dataset, 'valid', all_values)
    all_values, test_graphs = read_data(dataset, 'test', all_values)

    # get one-hot encoder for attribute values
    feature_encoder, n_values = get_feature_encoder(all_values)

    # save n_values to disk
    print('saving n_values to disk...')
    with open('../data/' + dataset + '/cache/n_values.txt', 'w') as f:
        f.writelines('%d' % n_values)

    return feature_encoder, n_values, train_graphs, valid_graphs, test_graphs


def load_data(model_args):
    # if datasets are already cached, skip preprocessing
    if os.path.exists('../data/' + model_args.dataset + '/cache/'):
        print('cache found')
        feature_encoder, train_graphs, valid_graphs, test_graphs = [None] * 4
        print('loading n_values from disk...')
        with open('../data/' + model_args.dataset + '/cache/n_values.txt') as f:
            n_values = int(f.readline())
    else:
        print('no cache found')
        os.mkdir('../data/' + model_args.dataset + '/cache/')
        feature_encoder, n_values, train_graphs, valid_graphs, test_graphs = preprocess(model_args.dataset)

    train_dataset = SmilesDataset(model_args, 'train', feature_encoder, train_graphs)
    valid_dataset = SmilesDataset(model_args, 'valid', feature_encoder, valid_graphs)
    test_dataset = SmilesDataset(model_args, 'test', feature_encoder, test_graphs)

    return n_values, train_dataset, valid_dataset, test_dataset
