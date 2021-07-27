import os
import dgl
import torch
import pickle
import pysmiles
from collections import defaultdict


attribute_names = ['element', 'charge', 'aromatic', 'hcount']


class SmilesDataset(dgl.data.DGLDataset):
    def __init__(self, args, mode, feature_encoder=None, raw_graphs=None):
        self.args = args
        self.mode = mode
        self.feature_encoder = feature_encoder
        self.raw_graphs = raw_graphs
        self.path = '../data/' + self.args.dataset + '/cache/' + self.mode
        self.reactant_graphs = []
        self.product_graphs = []
        super().__init__(name='Smiles_' + mode)

    def to_gpu(self):
        if torch.cuda.is_available():
            print('moving ' + self.mode + ' data to GPU')
            self.reactant_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.reactant_graphs]
            self.product_graphs = [graph.to('cuda:' + str(self.args.gpu)) for graph in self.product_graphs]

    def save(self):
        print('saving ' + self.mode + ' reactants to ' + self.path + '_reactant_graphs.bin')
        print('saving ' + self.mode + ' products to ' + self.path + '_product_graphs.bin')
        dgl.save_graphs(self.path + '_reactant_graphs.bin', self.reactant_graphs)
        dgl.save_graphs(self.path + '_product_graphs.bin', self.product_graphs)

    def load(self):
        print('loading ' + self.mode + ' reactants from ' + self.path + '_reactant_graphs.bin')
        print('loading ' + self.mode + ' products from ' + self.path + '_product_graphs.bin')
        # graphs loaded from disk will have a default empty label set: [graphs, labels], so we only take the first item
        self.reactant_graphs = dgl.load_graphs(self.path + '_reactant_graphs.bin')[0]
        self.product_graphs = dgl.load_graphs(self.path + '_product_graphs.bin')[0]
        self.to_gpu()

    def process(self):
        print('transforming ' + self.mode + ' data from networkx graphs to DGL graphs')
        for i, (raw_reactant_graph, raw_product_graph) in enumerate(self.raw_graphs):
            if i % 10000 == 0:
                print('%dk' % (i // 1000))
            # transform networkx graphs to dgl graphs
            reactant_graph = networkx_to_dgl(raw_reactant_graph, self.feature_encoder)
            product_graph = networkx_to_dgl(raw_product_graph, self.feature_encoder)
            self.reactant_graphs.append(reactant_graph)
            self.product_graphs.append(product_graph)
        self.to_gpu()

    def has_cache(self):
        return os.path.exists(self.path + '_reactant_graphs.bin') and os.path.exists(self.path + '_product_graphs.bin')

    def __getitem__(self, i):
        return self.reactant_graphs[i], self.product_graphs[i]

    def __len__(self):
        return len(self.reactant_graphs)


def networkx_to_dgl(raw_graph, feature_encoder):
    # add edges
    src = [s for (s, _) in raw_graph.edges]
    dst = [t for (_, t) in raw_graph.edges]
    graph = dgl.graph((src, dst), num_nodes=len(raw_graph.nodes))
    # add node features
    node_features = []
    for i in range(len(raw_graph.nodes)):
        raw_feature = raw_graph.nodes[i]
        numerical_feature = []
        for j in attribute_names:
            if raw_feature[j] in feature_encoder[j]:
                numerical_feature.append(feature_encoder[j][raw_feature[j]])
            else:
                numerical_feature.append(feature_encoder[j]['unknown'])
        node_features.append(numerical_feature)
    node_features = torch.tensor(node_features)
    graph.ndata['feature'] = node_features
    # transform to bi-directed graph with self-loops
    graph = dgl.to_bidirected(graph, copy_ndata=True)
    graph = dgl.add_self_loop(graph)
    return graph


def read_data(dataset, mode):
    path = '../data/' + dataset + '/' + mode + '.csv'
    print('preprocessing %s data from %s' % (mode, path))

    # saving all possible values of each attribute (only for training data)
    all_values = defaultdict(set)
    graphs = []

    with open(path) as f:
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

            if mode == 'train':
                # store all values
                for graph in [reactant_graph, product_graph]:
                    for attr in attribute_names:
                        for _, value in graph.nodes(data=attr):
                            all_values[attr].add(value)

            graphs.append([reactant_graph, product_graph])

    if mode == 'train':
        return all_values, graphs
    else:
        return graphs


def get_feature_encoder(all_values):
    feature_encoder = {}
    idx = 0
    # key: attribute; values: all possible values of the attribute
    for key, values in all_values.items():
        feature_encoder[key] = {}
        for value in values:
            feature_encoder[key][value] = idx
            idx += 1
        # for each attribute, we add an "unknown" key to handle unknown values during inference
        feature_encoder[key]['unknown'] = idx
        idx += 1

    return feature_encoder


def preprocess(dataset):
    print('preprocessing %s dataset' % dataset)

    # read all data and get all values for attributes
    all_values, train_graphs = read_data(dataset, 'train')
    valid_graphs = read_data(dataset, 'valid')
    test_graphs = read_data(dataset, 'test')

    # get one-hot encoder for attribute values
    feature_encoder = get_feature_encoder(all_values)

    # save feature encoder to disk
    path = '../data/' + dataset + '/cache/feature_encoder.pkl'
    print('saving feature encoder to %s' % path)
    with open(path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    return feature_encoder, train_graphs, valid_graphs, test_graphs


def load_data(args):
    # if datasets are already cached, skip preprocessing
    if os.path.exists('../data/' + args.dataset + '/cache/'):
        path = '../data/' + args.dataset + '/cache/feature_encoder.pkl'
        print('cache found\nloading feature encoder from %s' % path)
        with open(path, 'rb') as f:
            feature_encoder = pickle.load(f)
        train_dataset = SmilesDataset(args, 'train')
        valid_dataset = SmilesDataset(args, 'valid')
        test_dataset = SmilesDataset(args, 'test')
    else:
        print('no cache found')
        path = '../data/' + args.dataset + '/cache/'
        print('creating directory: %s' % path)
        os.mkdir(path)
        feature_encoder, train_graphs, valid_graphs, test_graphs = preprocess(args.dataset)
        train_dataset = SmilesDataset(args, 'train', feature_encoder, train_graphs)
        valid_dataset = SmilesDataset(args, 'valid', feature_encoder, valid_graphs)
        test_dataset = SmilesDataset(args, 'test', feature_encoder, test_graphs)

    return feature_encoder, train_dataset, valid_dataset, test_dataset
