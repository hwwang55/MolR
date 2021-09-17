import os
import argparse
import data_processing
import train
from property_pred import pp_data_processing, pp_train
from ged_pred import gp_data_processing, gp_train
from visualization import visualize


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    #'''
    # pretraining / chemical reaction prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=False, help='save the trained model to disk')
    #'''

    '''
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    '''

    '''
    # GED prediction
    parser.add_argument('--task', type=str, default='ged_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='QM9', help='dataset name')
    parser.add_argument('--n_molecules', type=int, default=1000, help='the number of molecules to be sampled')
    parser.add_argument('--n_pairs', type=int, default=10000, help='the number of molecule pairs to be sampled')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    parser.add_argument('--feature_mode', type=str, default='concat', help='how to construct the input feature')
    '''

    '''
    # visualization
    parser.add_argument('--task', type=str, default='visualization', help='downstream task')
    parser.add_argument('--subtask', type=str, default='size', help='subtask type: reaction, property, ged, size, ring')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size for calling the pretrained model')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    '''

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        data = data_processing.load_data(args)
        train.train(args, data)
    elif args.task == 'property_pred':
        data = pp_data_processing.load_data(args)
        pp_train.train(args, data)
    elif args.task == 'ged_pred':
        data = gp_data_processing.load_data(args)
        gp_train.train(args, data)
    elif args.task == 'visualization':
        visualize.draw(args)
    else:
        raise ValueError('unknown task')


if __name__ == '__main__':
    main()
