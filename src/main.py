import os
import argparse
import data_processing
import train
from property_pred import pp_data_processing, pp_train


def print_setting(args):
    print('\n===========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the index of gpu device')

    '''
    # pretraining or chemical reaction prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch', type=int, default=4096, help='batch size')
    parser.add_argument('--gnn', type=str, default='tag', help='name of the GNN model')
    parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--margin', type=float, default=4.0, help='margin in contrastive loss')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--save_model', type=bool, default=False, help='save the trained model to disk')
    '''

    #'''
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_pred', help='downstream task')
    parser.add_argument('--pretrained_model', type=str, default='gcn_1024', help='the pretrained model')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    parser.add_argument('--finetune', type=bool, default=False, help='whether fine-tuning the pretrained model')
    parser.add_argument('--pred_model', type=str, default='lr', help='downstream prediction model if not fine-tune')
    parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs if fine-tune')
    parser.add_argument('--batch', type=int, default=128, help='batch size if fine-tune')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate if fine-tune')
    #'''

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')

    if args.task == 'pretrain':
        data = data_processing.load_data(args)
        train.train(args, data)
    elif args.task == 'property_pred':
        data = pp_data_processing.load_data(args)
        pp_train.train(args, data)


if __name__ == '__main__':
    main()
