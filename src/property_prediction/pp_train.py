import torch
import pickle
from model import GNN
from dgl.dataloading import GraphDataLoader
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score


def train(args, data):
    print('loading hyperparameters of MolE from disk...')
    with open('../saved/' + args.hp_file, 'rb') as f:
        hp = pickle.load(f)

    print('building the MolE model...')
    mole = GNN(hp['gnn'], hp['n_layer'], hp['n_values'], hp['dim'], hp['dist_metric'])
    mole.load_state_dict(torch.load('../saved/' + args.model_file))
    if torch.cuda.is_available():
        mole = mole.cuda(args.gpu)

    if args.finetune:
        pass
    else:
        dataloader = GraphDataLoader(data, batch_size=args.batch_size, shuffle=True)
        all_features = []
        all_labels = []
        with torch.no_grad():
            mole.eval()
            for graphs, labels in dataloader:
                graph_embeddings = mole(graphs, graphs.ndata['feature'])
                all_features.append(graph_embeddings)
                all_labels.append(labels)
            all_features = torch.cat(all_features, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

        print('splitting dataset...')
        train_features = all_features[: int(0.8 * len(all_labels))]
        train_labels = all_labels[: int(0.8 * len(all_labels))]
        test_features = all_features[int(0.9 * len(all_labels)):]
        test_labels = all_labels[int(0.9 * len(all_labels)):]

        print('training the classification model...')
        if args.pred_model == 'svm':
            pred_model = SVC(probability=True)
        elif args.pred_model == 'lr':
            pred_model = LogisticRegression(solver='liblinear')
        elif args.pred_model == 'dt':
            pred_model = DecisionTreeClassifier()
        elif args.pred_model == 'mlp':
            pred_model = MLPClassifier()
        else:
            raise ValueError('unknown classification model')
        pred_model.fit(train_features, train_labels)
        acc = pred_model.score(test_features, test_labels)
        auc = roc_auc_score(test_labels, pred_model.predict_proba(test_features)[:, 1])
        print('\ntest acc: %.4f\ntest auc: %.4f' % (acc, auc))
