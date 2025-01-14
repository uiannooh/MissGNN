import argparse
import os
import optuna

import numpy as np
import torch

from training.gnn_mdi import train_gnn_mdi
from training.gnn_y import train_gnn_y
from uci.uci_subparser import add_uci_subparser
from utils.utils import auto_select_gpu

def main():
    study = optuna.create_study()
    study.optimize(objective, n_trials = 1)
    print('study.best_params:', study.best_params)
    # objective()

def objective(trial):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='EGSAGE_EGSAGE_EGSAGE')
    parser.add_argument('--post_hiddens', type=str, default=None,) # default to be 1 hidden of node_dim
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) # default to be all true
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--edge_dim', type=int, default=64)
    parser.add_argument('--edge_mode', type=int, default=1)  # 0: use it as weight; 1: as input to mlp
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--impute_hiddens', type=str, default='64')
    parser.add_argument('--impute_activation', type=str, default='relu')
    parser.add_argument('--predict_hiddens', type=str, default='')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='none')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    # parser.add_argument('--loss_mode', type=int, default = 0) # 0: loss on all train edge, 1: loss only on unknown train edge
    parser.add_argument('--valid', type=float, default=0.1) # valid-set ratio
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mdi_dir', type = str, default='mdi_ckpt')
    parser.add_argument('--y_dir', type = str, default='y_ckpt')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='train') # debug
    parser.add_argument('--pre_train', action='store_true', default=False)
    parser.add_argument('--task', type = str, default='reg')
    # parser.add_argument('--method', type = str, default='')
    # parser.add_argument('--process', type = str, default='mdi')
    
    subparsers = parser.add_subparsers()
    add_uci_subparser(subparsers)
    args = parser.parse_args()
    print(args)

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    params = {
        "epochs_mdi": 1800,
        "epochs_y": 1800,
        "lr": 0.001,
        "n_subsets": 4,
        "overlap": 0.75,
        "model_types": "EGSAGE_EGSAGE_EGSAGE",
        "post_hiddens": "64",
        "node_dim": 128,
        "edge_dim": 96,
        "impute_hiddens": "64",
        "predict_hiddens": "256"
    }

    from uci.uci_data import load_data

    mdi_path = './{}/test/{}/{}/'.format(args.domain, args.data, args.mdi_dir)
    y_path = './{}/test/{}/{}/'.format(args.domain, args.data, args.y_dir)

    try: os.makedirs(mdi_path)
    except: os.remove(mdi_path + 'mdi_model.pt')

    try: os.makedirs(y_path)
    except: os.remove(y_path + 'y_model.pt')
    
    print('data:', args.data)

    l1_all = 0
    for seed in range(1, 5):
        args.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        print('seed', seed)
        data = load_data(args, params, device)

        train_gnn_mdi(data, args, params, mdi_path, device)
        print("--------------- mdi over, start prediction ---------------")
        l1 = train_gnn_y(data, args, params, mdi_path, y_path, device)

        l1_all += l1

    print('data:', args.data)

    return l1_all / 5


if __name__ == '__main__':
    main()