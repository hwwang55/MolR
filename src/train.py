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
    model = GNN(args.gnn, args.layer, n_values, args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batchsize, shuffle=True, drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda(args.gpu)

    best_model_params = None
    best_val_mrr = 0
    print('start training\n')

    print('initial case:')
    model.eval()
    evaluate(model, 'valid', valid_data, args)
    print()

    for i in range(args.epoch):
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
        val_mrr = evaluate(model, 'valid', valid_data, args)

        # save the best model
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            best_model_params = deepcopy(model.state_dict())

        print()

    # evaluation on the test set
    print('final results on the test set:')
    model.load_state_dict(best_model_params)
    evaluate(model, 'test', test_data, args)
    
    # save the model， hyperparameters, and feature encoder to disk
    if not os.path.exists('../saved/'):
        print('creating directory: ../saved/')
        os.mkdir('../saved/')
    prefix = time.strftime('%Y%m%d%H%M%S', time.localtime())

    model_path = '../saved/%s_model.pt' % prefix
    print('\nsaving the trained model to %s' % model_path)
    torch.save(best_model_params, model_path)

    fe_path = '../saved/%s_feature_encoder.pkl' % prefix
    print('saving feature encoder to %s' % fe_path)
    with open(fe_path, 'wb') as f:
        pickle.dump(feature_encoder, f)

    hp_path = '../saved/%s_hyperparameters.pkl' % prefix
    print('saving hyperparameters to %s' % hp_path)
    with open(hp_path, 'wb') as f:
        hp_dict = {'gnn': args.gnn, 'layer': args.layer, 'n_values': n_values, 'dim': args.dim}
        pickle.dump(hp_dict, f)


def calculate_loss(reactant_embeddings, product_embeddings, args):
    dist = torch.cdist(reactant_embeddings, product_embeddings, p=2)
    pos = torch.diag(dist)
    mask = torch.eye(args.batchsize)
    if torch.cuda.is_available():
        mask = mask.cuda(args.gpu)
    neg = (1 - mask) * dist + mask * args.margin
    neg = torch.relu(args.margin - neg)
    loss = torch.mean(pos) + torch.sum(neg) / args.batchsize / (args.batchsize - 1)
    return loss


def evaluate(model, mode, data, args):
    model.eval()
    with torch.no_grad():
        # calculate embeddings of all products as the candidate pool
        product_dataloader = GraphDataLoader(data, batch_size=args.batchsize)
        all_product_embeddings = []
        for _, product_graphs in product_dataloader:
            product_embeddings = model(product_graphs)
            all_product_embeddings.append(product_embeddings)
        all_product_embeddings = torch.cat(all_product_embeddings, dim=0)

        # rank
        all_rankings = []
        reactant_dataloader = GraphDataLoader(data, batch_size=args.batchsize)
        i = 0
        for reactant_graphs, _ in reactant_dataloader:
            reactant_embeddings = model(reactant_graphs)
            ground_truth = torch.unsqueeze(torch.arange(i, min(i + args.batchsize, len(data))), dim=1)
            i += args.batchsize
            if torch.cuda.is_available():
                ground_truth = ground_truth.cuda(args.gpu)
            dist = torch.cdist(reactant_embeddings, all_product_embeddings, p=2)
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
