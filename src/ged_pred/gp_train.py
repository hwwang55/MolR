import torch
import pickle
from model import GNN
from dgl.dataloading import GraphDataLoader
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train(args, data):
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

    dataloader = GraphDataLoader(data, batch_size=args.batch_size, shuffle=True)
    all_features1 = []
    all_features2 = []
    all_labels = []
    with torch.no_grad():
        mole.eval()
        for graphs1, graphs2, labels in dataloader:
            graph_embeddings1 = mole(graphs1)
            graph_embeddings2 = mole(graphs2)
            all_features1.append(graph_embeddings1)
            all_features2.append(graph_embeddings2)
            all_labels.append(labels)
        all_features1 = torch.cat(all_features1, dim=0)
        all_features2 = torch.cat(all_features2, dim=0)
        if args.feature_mode == 'concat':
            all_features = torch.cat([all_features1, all_features2], dim=-1).cpu().numpy()
        elif args.feature_mode == 'subtract':
            all_features = (all_features1 - all_features2).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    print('splitting dataset')
    train_features = all_features[: int(0.8 * len(data))]
    train_labels = all_labels[: int(0.8 * len(data))]
    valid_features = all_features[int(0.8 * len(data)): int(0.9 * len(data))]
    valid_labels = all_labels[int(0.8 * len(data)): int(0.9 * len(data))]
    test_features = all_features[int(0.9 * len(data)):]
    test_labels = all_labels[int(0.9 * len(data)):]

    print('training the regression model\n')
    pred_model = SVR()
    pred_model.fit(train_features, train_labels)
    run_regression(pred_model, 'train', train_features, train_labels)
    run_regression(pred_model, 'valid', valid_features, valid_labels)
    run_regression(pred_model, 'test', test_features, test_labels)


def run_regression(model, mode, features, labels):
    pred = model.predict(features)
    mae = mean_absolute_error(labels, pred)
    rmse = mean_squared_error(labels, pred, squared=False)
    print('%s mae: %.4f   rmse: %.4f' % (mode, mae, rmse))
