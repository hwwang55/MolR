import os
import time
import torch
import pickle
import dataloader
import numpy as np
from model import GNN
from copy import deepcopy
from dgl.dataloading import GraphDataLoader

# torch.set_printoptions(profile="full", linewidth=100000, sci_mode=False)


def train(args, data):
    feature_encoder, train_data, valid_data, test_data = data
    n_values = sum([len(feature_encoder[key]) for key in dataloader.attribute_names])
    model = GNN(args.gnn, args.n_layer, n_values, args.dim, args.dist_metric)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    print('start training...\n')

    print('initial case:')
    model.eval()
    evaluate(model, 'valid', valid_data, args)
    print()

    for i in range(args.n_epoch):
        print('epoch %d:' % i)

        # train
        model.train()
        for reactant_graphs, product_graphs in train_dataloader:
            reactant_embeddings = model(reactant_graphs)
            product_embeddings = model(product_graphs)
            loss = calculate_loss(reactant_embeddings, product_embeddings, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate on the validation set
        model.eval()
        val_mrr = evaluate(model, 'valid', valid_data, args)

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())
        print()

    # evaluation on the test set
    print('evaluating on the test set...')
    model.load_state_dict(best_model_params)
    model.eval()
    evaluate(model, 'test', test_data, args)

    '''
    # save the model and the feature encoder to disk
    print('\nsaving the trained model, hyperparameters, and the feature encoder to disk...')
    if not os.path.exists('../saved/'):
        os.mkdir('../saved/')
    prefix = time.strftime('%Y%m%d%H%M%S', time.localtime())
    torch.save(best_model_params, '../saved/%s_model.pt' % prefix)
    with open('../saved/%s_fe.pkl' % prefix, 'wb') as f:
        pickle.dump(feature_encoder, f)
    with open('../saved/%s_hp.pkl' % prefix, 'wb') as f:
        hp_dict = {'gnn': args.gnn, 'n_layer': args.n_layer, 'n_values': n_values, 'dim': args.dim,
                   'dist_metric': args.dist_metric}
        pickle.dump(hp_dict, f)
    '''


def calculate_loss(reactant_embeddings, product_embeddings, args):
    if args.dist_metric == 'euclidean':
        dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
        pos = torch.diag(dist)
        mask = torch.eye(args.batch_size)
        if torch.cuda.is_available():
            mask = mask.cuda(args.gpu)
        neg = (1 - mask) * dist + mask * args.margin
        neg = torch.relu(args.margin - neg)
        loss = torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)
    elif args.dist_metric == 'dot':
        dist = torch.matmul(reactant_embeddings, torch.transpose(product_embeddings, 0, 1))
        pos = torch.diag(dist)
        mask = torch.eye(args.batch_size)
        if torch.cuda.is_available():
            mask = mask.cuda(args.gpu)
        neg = (1 - mask) * dist
        loss = -torch.mean(pos) + torch.sum(neg) / args.batch_size / (args.batch_size - 1)
    else:
        raise ValueError('unknown distance metric')
    return loss


def evaluate(model, mode, data, args):
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        product_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        all_product_embeddings = []
        for _, product_graphs in product_dataloader:
            product_embeddings = model(product_graphs)
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        # rank
        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=args.batch_size)
        i = 0
        for reactant_graphs, _ in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batch_size, len(data))), dim=1)
            i += args.batch_size
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.gpu)
            if args.dist_metric == 'euclidean':
                dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
            elif args.dist_metric == 'dot':
                dist = -torch.matmul(reactant_embeddings, torch.transpose(all_product_embeddings, 0, 1))
            else:
                raise ValueError('unknown distance metric')
            sorted_indices = torch.argsort(dist, dim=1)
            rankings = ((sorted_indices == ground_truth).nonzero()[:, 1] + 1).tolist()
            all_rankings.extend(rankings)

        # calculate metrics
        all_rankings = np.array(all_rankings)
        mrr = float(np.mean(1 / all_rankings))
        mr = float(np.mean(all_rankings))
        h1 = float(np.mean(all_rankings <= 1))
        h3 = float(np.mean(all_rankings <= 3))
        h5 = float(np.mean(all_rankings <= 5))
        h10 = float(np.mean(all_rankings <= 10))

        print('%s  mrr: %.4f  mr: %.4f  h1: %.4f  h3: %.4f  h5: %.4f  h10: %.4f' % (mode, mrr, mr, h1, h3, h5, h10))
        return mrr
