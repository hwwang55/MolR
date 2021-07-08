import torch
import pickle
import numpy as np
from model import GNN
from property_prediction.pp_model import PropertyPredictionModel
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import BCEWithLogitsLoss
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

    if args.finetune:
        train_with_finetune(args, data, mole)
    else:
        train_without_finetune(args, data, mole)


def train_with_finetune(args, data, mole):
    pred_model = PropertyPredictionModel(mole)
    if torch.cuda.is_available():
        pred_model = pred_model.cuda(args.gpu)
    optimizer = torch.optim.Adam(pred_model.parameters(), lr=args.lr, weight_decay=args.l2)
    loss_fn = BCEWithLogitsLoss()
    train_indices = np.random.choice(range(len(data)), size=int(0.8 * len(data)), replace=False)
    left = set(range(len(data))) - set(train_indices)
    valid_indices = np.random.choice(list(left), size=int(0.1 * len(data)), replace=False)
    test_indices = list(left - set(valid_indices))
    train_dataloader = GraphDataLoader(data, sampler=SubsetRandomSampler(train_indices), batch_size=args.batch_size)
    valid_dataloader = GraphDataLoader(data, sampler=SubsetRandomSampler(valid_indices), batch_size=args.batch_size)
    test_dataloader = GraphDataLoader(data, sampler=SubsetRandomSampler(test_indices), batch_size=args.batch_size)

    best_valid_acc = 0
    best_test_acc = 0
    best_test_auc = 0
    print('start training...\n')

    for i in range(args.n_epoch):
        print('epoch %d:' % i)

        # train
        pred_model.train()
        for graphs, labels in train_dataloader:
            pred = torch.squeeze(pred_model(graphs))
            loss = loss_fn(input=pred, target=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        pred_model.eval()
        train_acc, train_auc = evaluate(pred_model, 'train', train_dataloader)
        valid_acc, valid_auc = evaluate(pred_model, 'valid', valid_dataloader)
        test_acc, test_auc = evaluate(pred_model, 'test', test_dataloader)

        # save the best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_test_acc = test_acc
            best_test_auc = test_auc
        print()

    print('final test acc: %.4f    auc: %.4f' % (best_test_acc, best_test_auc))


def evaluate(model, mode, dataloader):
    all_proba = []
    all_labels = []
    with torch.no_grad():
        for graphs, labels in dataloader:
            proba = torch.squeeze(torch.sigmoid(model(graphs))).tolist()
            all_proba.extend(proba)
            all_labels.extend(labels.tolist())
    pred = [1.0 if p > 0.5 else 0.0 for p in all_proba]
    acc = float(np.mean(np.array(pred) == np.array(all_labels)))
    auc = roc_auc_score(all_labels, all_proba)
    print('%s acc: %.4f    auc: %.4f' % (mode, acc, auc))
    return acc, auc


def train_without_finetune(args, data, mole):
    if torch.cuda.is_available():
        mole = mole.cuda(args.gpu)

    dataloader = GraphDataLoader(data, batch_size=args.batch_size, shuffle=True)
    all_features = []
    all_labels = []
    with torch.no_grad():
        mole.eval()
        for graphs, labels in dataloader:
            graph_embeddings = mole(graphs)
            all_features.append(graph_embeddings)
            all_labels.append(labels)
        all_features = torch.cat(all_features, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    print('splitting dataset...')
    train_features = all_features[: int(0.8 * len(data))]
    train_labels = all_labels[: int(0.8 * len(data))]
    test_features = all_features[int(0.9 * len(data)):]
    test_labels = all_labels[int(0.9 * len(data)):]

    print('training the classification model...')
    if args.pred_model == 'svm':
        pred_model = SVC(probability=True)
    elif args.pred_model == 'lr':
        pred_model = LogisticRegression()
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
