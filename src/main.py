import argparse
from data_loader import load_data
from train import train


def print_setting(args):
    print('\n========================')
    print('dataset: ' + args.dataset)
    print('n_epoch: ' + str(args.n_epoch))
    print('batch_size: ' + str(args.batch_size))
    print('gnn: ' + args.gnn)
    print('n_layer: ' + str(args.n_layer))
    print('dim: ' + str(args.dim))
    print('dist_metric: ' + args.dist_metric)
    print('margin: ' + str(args.margin))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('========================\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset the model is pretrained on')
    parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--n_layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of embeddings')
    parser.add_argument('--dist_metric', type=str, default='euclidean', help='distance metric of molecule embeddings')
    parser.add_argument('--margin', type=float, default=64, help='the margin in contrastive loss')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    parser.add_argument('--gpu', type=int, default=2, help='the index of gpu device')

    args = parser.parse_args()
    print_setting(args)
    data = load_data(args)
    train(args, data)


if __name__ == '__main__':
    main()
