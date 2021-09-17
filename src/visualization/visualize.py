import os
import torch
import pickle
import pysmiles
import matplotlib
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from model import GNN
from openbabel import pybel
from featurizer import MolEFeaturizer
from dgl.dataloading import GraphDataLoader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from networkx.algorithms.similarity import graph_edit_distance
from property_pred.pp_data_processing import PropertyPredDataset

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.figure(figsize=(7, 4))


query_smiles = 'C3=C(C2=CC=C(N1CCC(O)CC1)N=N2)C(=CC(=C3)Cl)Cl'  # num 1196 molecule in BBBP dataset
query_no = 1196
query_graph = pysmiles.read_smiles(query_smiles, zero_order_bonds=False)
upper_bound = 50
timeout = 300


def get_sssr(args):
    if os.path.exists('../data/' + args.dataset + '/sssr.pkl'):
        print('loading GED data from ../data/' + args.dataset + '/sssr.pkl')
        with open('../data/' + args.dataset + '/sssr.pkl', 'rb') as f:
            res = pickle.load(f)
    else:
        smiles_list = []
        print('processing ' + '../data/' + args.dataset + '/' + args.dataset + '.csv')
        with open('../data/' + args.dataset + '/' + args.dataset + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                items = line.strip().split(',')
                if args.dataset == 'BBBP':
                    smiles = items[-1]
                    pysmiles.read_smiles(smiles)
                else:
                    raise ValueError('unknown dataset')
                smiles_list.append(smiles)
        res = [len(pybel.readstring('smi', s).OBMol.GetSSSR()) for s in smiles_list]

        print('saving SSSR data to ../data/' + args.dataset + '/sssr.pkl')
        with open('../data/' + args.dataset + '/sssr.pkl', 'wb') as f:
            pickle.dump(res, f)

    return res


def get_ged(args):
    if os.path.exists('../data/' + args.dataset + '/ged_wrt_' + str(query_no) + '.pkl'):
        print('loading GED data from ../data/' + args.dataset + '/ged_wrt_' + str(query_no) + '.pkl')
        with open('../data/' + args.dataset + '/ged_wrt_' + str(query_no) + '.pkl', 'rb') as f:
            res = pickle.load(f)
    else:
        smiles_list = []
        print('processing ' + '../data/' + args.dataset + '/' + args.dataset + '.csv')
        with open('../data/' + args.dataset + '/' + args.dataset + '.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue
                items = line.strip().split(',')
                if args.dataset == 'BBBP':
                    smiles = items[-1]
                    pysmiles.read_smiles(smiles)
                else:
                    raise ValueError('unknown dataset')
                smiles_list.append(smiles)
        smiles2ged = calculate_ged_with_mp(smiles_list)
        res = [smiles2ged[s] for s in smiles_list]

        print('saving GED data to ../data/' + args.dataset + '/ged_wrt_' + str(query_no) + '.pkl')
        with open('../data/' + args.dataset + '/ged_wrt_' + str(query_no) + '.pkl', 'wb') as f:
            pickle.dump(res, f)

    return res


def calculate_ged_with_mp(smiles_list):
    print('calculating GED using multiprocessing')
    n_cores, pool, range_list = get_params_for_mp(len(smiles_list))
    dict_list = pool.map(calculate_ged, zip([smiles_list[i[0]: i[1]] for i in range_list], range(n_cores)))
    print('gathering results')
    res = {}
    for d in dict_list:
        res.update(d)
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

    res = {}
    smiles_list, pid = inputs
    for i, smiles in enumerate(smiles_list):
        graph = pysmiles.read_smiles(smiles, zero_order_bonds=False)
        ged = graph_edit_distance(
            graph, query_graph, node_match=node_match, edge_match=edge_match, upper_bound=upper_bound, timeout=timeout)
        res[smiles] = ged
        print('pid %d:  %d / %d' % (pid, i, len(smiles_list)))
    print('pid %d  done' % pid)
    return res


def draw(args):
    if args.subtask == 'reaction':
        model = MolEFeaturizer('../saved/' + args.pretrained_model)
        emb, _ = model.transform(['CCO', 'CC=O', 'CC(=O)-O',
                                  'CCCCCCCCO', 'CCCCCCCC=O', 'CCCCCCCC(=O)O',
                                  'OCCO', 'O=CC=O', 'OC(=O)C(=O)O'
                                  ])
        emb = PCA(n_components=2).fit_transform(emb)
        color = ['red', 'darkorange', 'blue']
        plt.plot(emb[0, 0], emb[0, 1], marker='o', color='red', markerfacecolor='none', markersize=8)
        plt.plot(emb[1, 0], emb[1, 1], marker='^', color='red', markerfacecolor='none', markersize=8)
        plt.plot(emb[2, 0], emb[2, 1], marker='s', color='red', markerfacecolor='none', markersize=8)
        plt.plot(emb[3, 0], emb[3, 1], marker='o', color='darkorange', markerfacecolor='none', markersize=8)
        plt.plot(emb[4, 0], emb[4, 1], marker='^', color='darkorange', markerfacecolor='none', markersize=8)
        plt.plot(emb[5, 0], emb[5, 1], marker='s', color='darkorange', markerfacecolor='none', markersize=8)
        plt.plot(emb[6, 0], emb[6, 1], marker='o', color='blue', markerfacecolor='none', markersize=8)
        plt.plot(emb[7, 0], emb[7, 1], marker='^', color='blue', markerfacecolor='none', markersize=8)
        plt.plot(emb[8, 0], emb[8, 1], marker='s', color='blue', markerfacecolor='none', markersize=8)
        plt.show()
        # plt.savefig('visualization/' + args.subtask + '.pdf', bbox_inches='tight')
    else:
        data = PropertyPredDataset(args)
        path = '../saved/' + args.pretrained_model + '/'
        print('loading hyperparameters of pretrained model from ' + path + 'hparams.pkl')
        with open(path + 'hparams.pkl', 'rb') as f:
            hparams = pickle.load(f)

        print('loading pretrained model from ' + path + 'model.pt')
        mole = GNN(hparams['gnn'], hparams['layer'], hparams['feature_len'], hparams['dim'])
        if torch.cuda.is_available():
            mole.load_state_dict(torch.load(path + 'model.pt'))
            mole = mole.cuda(args.gpu)
        else:
            mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))

        dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        emb = []
        properties = []
        with torch.no_grad():
            mole.eval()
            for graphs_batch, labels_batch in dataloader:
                embeddings_batch = mole(graphs_batch)
                emb.append(embeddings_batch)
                properties.append(labels_batch)
            emb = torch.cat(emb, dim=0).cpu().numpy()
            properties = torch.cat(properties, dim=0).cpu().numpy()

        if args.subtask == 'size':
            n_quantiles = 4
            sizes = [g.num_nodes() for g in data.graphs]
            thresholds = [np.quantile(sizes, i / n_quantiles) for i in range(1, n_quantiles)]
            labels = np.zeros_like(sizes)
            for i, q in enumerate(thresholds):
                labels[sizes >= q] = i + 1
            legend = [r'1 $\leq$ size $<$ 18', r'18 $\leq$ size $<$ 23', r'23 $\leq$ size $<$ 28', r'28 $\leq$ size']
            colors = ['lightskyblue', 'gold', 'darkorange', 'maroon']
        elif args.subtask == 'property':
            labels = properties
            thresholds = [0.5]
            legend = ['non-permeable', 'permeable']
            colors = ['maroon', 'gold']
        elif args.subtask == 'ged':
            ged = get_ged(args)
            ged = np.array([d if d is not None else upper_bound + 10 for d in ged])
            thresholds = [30, 50]
            labels = np.zeros_like(ged)
            for i, q in enumerate(thresholds):
                labels[ged >= q] = i + 1
            legend = [r'1 $\leq$ GED $<$ 30', r'30 $\leq$ GED $<$ 50', r'50 $\leq$ GED']
            colors = ['darkorange', 'lightskyblue', 'maroon']
        elif args.subtask == 'ring':
            ring_cnt = np.array(get_sssr(args))
            thresholds = [1, 2, 3]
            labels = np.zeros_like(ring_cnt)
            for i, q in enumerate(thresholds):
                labels[ring_cnt >= q] = i + 1
            legend = [r'# rings $=$ 0', r'# rings $=$ 1', r'# rings $=$ 2', r'# rings $\geq$ 3']
            colors = ['lightskyblue', 'gold', 'darkorange', 'maroon']
        else:
            raise ValueError('unknown subtask')

        print('calculating TSNE embeddings')
        tsne = TSNE(random_state=0).fit_transform(emb)
        for i in range(len(thresholds) + 1):
            plt.scatter(tsne[labels == i, 0], tsne[labels == i, 1], s=3, c=colors[i])
        plt.legend(legend, loc='upper right', fontsize=9, ncol=1)
        plt.show()
        # plt.savefig('visualization/' + args.subtask + '.pdf', bbox_inches='tight')
