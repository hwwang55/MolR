import argparse
import dataloader
import train
from property_prediction import pp_dataloader, pp_train


def print_setting(args):
    print('\n========================')
    for k, v, in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2, help='the index of gpu device')

    '''
    # pretraining or chemical reaction prediction
    parser.add_argument('--task', type=str, default='pretrain', help='downstream task')
    parser.add_argument('--dataset', type=str, default='USPTO-479k', help='dataset name')
    parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model')
    parser.add_argument('--n_layer', type=int, default=2, help='number of GNN layers')
    parser.add_argument('--dim', type=int, default=1024, help='dimension of molecule embeddings')
    parser.add_argument('--dist_metric', type=str, default='euclidean', help='distance metric of molecule embeddings')
    parser.add_argument('--margin', type=float, default=64, help='the margin in contrastive loss')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    '''

    #'''
    # molecule property prediction
    parser.add_argument('--task', type=str, default='property_prediction', help='downstream task')
    parser.add_argument('--model_file', type=str, default='model.pt', help='filename of the pretrained model')
    parser.add_argument('--fe_file', type=str, default='fe.pkl', help='filename of the feature encoder')
    parser.add_argument('--hp_file', type=str, default='hp.pkl', help='filename of the hyperparameter dictionary')
    parser.add_argument('--dataset', type=str, default='BBBP', help='dataset name')
    parser.add_argument('--finetune', type=bool, default=False, help='whether fine-tuning the pretrained model')
    parser.add_argument('--pred_model', type=str, default='lr', help='downstream prediction model if not fine-tune')
    parser.add_argument('--n_epoch', type=int, default=20, help='number of epochs if fine-tune')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size if fine-tune')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight if fine-tune')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate if fine-tune')
    #'''

    args = parser.parse_args()
    print_setting(args)
    if args.task == 'pretrain':
        data = dataloader.load_data(args)
        train.train(args, data)
    elif args.task == 'property_prediction':
        data = pp_dataloader.load_data(args)
        pp_train.train(args, data)


if __name__ == '__main__':
    main()
