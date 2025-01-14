import pandas as pd
import os.path as osp
import inspect
from torch_geometric.data import Data
from sklearn import preprocessing
import itertools

import torch
import numpy as np

from utils.subsets_generator import subsets_generator
from utils.utils import get_known_mask, mask_edge

def create_node(df, mode):
    if mode == 0:  # onehot for feature nodes, all 1 for sample nodes
        row, col = df.shape
        feature_ind = np.array(range(col))
        feature_node = np.zeros((col, col))
        feature_node[np.arange(col), feature_ind] = 1
        sample_node = [[1]*col for i in range(row)]
        node = sample_node + feature_node.tolist()
    elif mode == 1:  # onehot for sample and feature nodes
        row, col = df.shape
        feature_ind = np.array(range(col))
        feature_node = np.zeros((col, col+1))
        feature_node[np.arange(col), feature_ind+1] = 1
        sample_node = np.zeros((row, col+1))
        sample_node[:, 0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node


def create_edge(df):
    row, col = df.shape
    edge_data = torch.arange(row).repeat_interleave(col)
    edge_feature = (torch.arange(col) + row).repeat(row)
    source = torch.cat([edge_data, edge_feature])
    target = torch.cat([edge_feature, edge_data])
    edge_index = torch.stack([source, target])
    return edge_index


def create_edge_attr(df):
    row, col = df.shape
    edge_attr = [[float(df.iloc[i,j])] for i in range(row) for j in range(col)]
    edge_attr = edge_attr + edge_attr
    return edge_attr


def to_comb(origin_list, axis, device = torch.device('cpu')):
    comb_list = list(itertools.combinations(origin_list, 2))  # len(origin_list) choose 2
    comb = [torch.cat([i, j], axis = axis).to(device) for i, j in comb_list]
    return comb, comb_list


def create_edge_index_comb(column_idx_list, column_idx_comb_list, row, col, device):
    # (a, b): which subset of the combination
    # (sample, feature): sample nodes / feature nodes
    edge_sample_a = torch.arange(row).repeat_interleave(len(column_idx_list[0])).expand(len(column_idx_comb_list), -1).to(device)
    edge_sample_b = torch.arange(row).repeat_interleave(len(column_idx_list[0])).expand(len(column_idx_comb_list), -1).to(device) + row + col

    edge_feature_a = []
    edge_feature_b = []
        
    for a, b in column_idx_comb_list:
        edge_feature_a.append(a.repeat(row) + row)
        edge_feature_b.append(b.repeat(row) + row * 2 + col)

    edge_index_comb = []
    for i in range(len(column_idx_comb_list)):
        # (source, target): the role it takes in terms of propagation
        source_a = torch.cat([edge_sample_a[i], edge_feature_a[i]])
        source_b = torch.cat([edge_sample_b[i], edge_feature_b[i]])
        target_a = torch.cat([edge_feature_a[i], edge_sample_a[i]])
        target_b = torch.cat([edge_feature_b[i], edge_sample_b[i]])
        
        source = torch.cat([source_a, source_b])
        target = torch.cat([target_a, target_b])
        
        edge_index_comb.append(torch.stack([source, target]))
    
    return edge_index_comb


def create_edge_index(column_idx_list, row, device):
    edge_sample = torch.arange(row).repeat_interleave(len(column_idx_list[0])).expand(len(column_idx_list), -1).to(device)
    
    edge_feature = []
    for column_idx in column_idx_list:
        edge_end_init = (column_idx + row).repeat(row)
        edge_feature.append(edge_end_init.to(device))
    
    edge_index = []
    for i in range(len(column_idx_list)):
        source = torch.cat([edge_sample[i], edge_feature[i]])
        target = torch.cat([edge_feature[i], edge_sample[i]])
        
        edge_index.append(torch.stack([source, target]))
    
    return edge_index


def data_preprocessing(df_X, args, params, device = torch.device('cpu')):
    column_idx_list = subsets_generator(df_X, args, params)
    column_idx_comb, column_idx_comb_list = to_comb(column_idx_list, axis = 0, device = device)
    
    x = torch.tensor(create_node(df_X, args.node_mode))
    x_comb = torch.cat([x, x]).to(device)

    edge_index_all = create_edge(df_X)

    row, col = df_X.shape
    edge_index_comb = create_edge_index_comb(column_idx_list, column_idx_comb_list, row, col, device)  # edge index for subset combinations
    edge_index = create_edge_index(column_idx_list, row, device)  # edge index for subsets
        
    df_X_sub = [df_X.iloc[:, column_idx.tolist()] for column_idx in column_idx_list]  # subsets of dataframe
    edge_attr = [torch.tensor(create_edge_attr(df_X_sub[i]), dtype = torch.float).to(device) for i in range(len(df_X_sub))]
    edge_attr_comb, _ = to_comb(edge_attr, axis = 0, device = device)

    return x_comb, edge_index_all, edge_index_comb, edge_attr_comb, column_idx_comb, edge_index, edge_attr, column_idx_list


def get_data(df_X, df_y, args, params, device, normalize = True):
    df_y = df_y[0].to_numpy()

    if normalize:
        x = df_X.values
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df_X = pd.DataFrame(x_scaled)
    x_comb, edge_index_all, edge_index_comb, edge_attr_comb, column_idx_comb, edge_index, edge_attr, column_idx_list \
        = data_preprocessing(df_X, args, params)
    node_init = create_node(df_X, args.node_mode) 
    x = torch.tensor(node_init, dtype = torch.float)
    y = torch.tensor(df_y, dtype = torch.float)
    
    # set seed to fix known / unknown edges
    torch.manual_seed(args.seed)

    # get comb data (for imputation)
    train_edge_index_comb = []
    train_edge_attr_comb = []
    train_labels_comb = []
    test_edge_index_comb = []
    test_edge_attr_comb = []
    test_labels_comb = []
    for i in range(len(column_idx_comb)):
        train_edge_mask = get_known_mask(args.train_edge, int(edge_attr_comb[i].shape[0]/4))
        quad_train_edge_mask = train_edge_mask.repeat(4)  # make sure missing data is missed on both direction of an edge
        train_edge_index_csub, train_edge_attr_csub = mask_edge(edge_index_comb[i], edge_attr_comb[i], quad_train_edge_mask, True)
        train_edge_index_comb.append(train_edge_index_csub.to(device))
        train_edge_attr_comb.append(train_edge_attr_csub.to(device))
        a, b = torch.split(train_edge_attr_csub, int(train_edge_attr_csub.shape[0]/2))  # attr of both subsets
        a = a[:int(a.shape[0]/2), 0]  # take only the first half,
        b = b[:int(b.shape[0]/2), 0]  # since both halves are the same
        train_labels_sub = torch.cat([a, b])
        train_labels_comb.append(train_labels_sub.to(device))

        test_edge_index_sub, test_edge_attr_sub = mask_edge(edge_index_comb[i], edge_attr_comb[i], ~quad_train_edge_mask, True)
        test_edge_index_comb.append(test_edge_index_sub.to(device))
        test_edge_attr_comb.append(test_edge_attr_sub.to(device))
        a, b = torch.split(test_edge_attr_sub, int(test_edge_attr_sub.shape[0]/2))
        a = a[:int(a.shape[0]/2), 0]
        b = b[:int(b.shape[0]/2), 0]
        test_labels_sub = torch.cat([a, b])
        test_labels_comb.append(test_labels_sub.to(device))

    #  get non-comb data (for prediction)
    train_edge_index = []
    train_edge_attr = []
    test_edge_index = []
    test_edge_attr = []

    train_edge_all_mask = get_known_mask(args.train_edge, int(edge_index_all.shape[1]/2))
    double_train_edge_all_mask = torch.cat([train_edge_all_mask, train_edge_all_mask])
    edge_index_to_drop = edge_index_all[:, ~double_train_edge_all_mask]

    for i in range(len(column_idx_list)):
        train_edge_mask = torch.ones(edge_index[i].shape[1], dtype = torch.bool)
        for j in range(edge_index_to_drop.shape[1]):
            drop_pair = edge_index_to_drop[:, j].unsqueeze(1)
            match = (edge_index[i] == drop_pair).all(axis = 0)
            train_edge_mask &= ~match
        train_edge_index_sub, train_edge_attr_sub = mask_edge(edge_index[i], edge_attr[i], train_edge_mask, True)
        train_edge_index.append(train_edge_index_sub.to(device))
        train_edge_attr.append(train_edge_attr_sub.to(device))
        test_edge_index_sub, test_edge_attr_sub = mask_edge(edge_index[i], edge_attr[i], ~train_edge_mask, True)
        test_edge_index.append(test_edge_index_sub.to(device))
        test_edge_attr.append(test_edge_attr_sub.to(device))

    # for i in range(len(column_idx_list)):
    #     train_edge_mask = get_known_mask(args.train_edge, int(edge_attr[i].shape[0]/2))
    #     double_train_edge_mask = train_edge_mask.repeat(2)

    #     train_edge_index_sub, train_edge_attr_sub = mask_edge(edge_index[i], edge_attr[i], double_train_edge_mask, True)
    #     train_edge_index.append(train_edge_index_sub.to(device))
    #     train_edge_attr.append(train_edge_attr_sub.to(device))
    #     test_edge_index_sub, test_edge_attr_sub = mask_edge(edge_index[i], edge_attr[i], ~double_train_edge_mask, True)
    #     test_edge_index.append(test_edge_index_sub.to(device))
    #     test_edge_attr.append(test_edge_attr_sub.to(device))


    # mask the y-values during training, i.e. train-test-split
    train_y_mask = get_known_mask(args.train_y, y.shape[0])
    test_y_mask = ~train_y_mask


    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr,
        x_comb=x_comb, edge_index_comb=edge_index_comb, edge_attr_comb=edge_attr_comb,
        train_y_mask=train_y_mask, test_y_mask=test_y_mask,
        train_edge_index_comb=train_edge_index_comb, train_edge_attr_comb=train_edge_attr_comb,
        train_labels_comb=train_labels_comb,
        test_edge_index_comb=test_edge_index_comb, test_edge_attr_comb=test_edge_attr_comb,
        test_labels_comb=test_labels_comb,
        edge_index_all=edge_index_all, 
        train_edge_index=train_edge_index, train_edge_attr=train_edge_attr,
        test_edge_index=test_edge_index, test_edge_attr=test_edge_attr,
        df_X=df_X, df_y=df_y,
        edge_attr_dim=1,
        column_idx_comb=column_idx_comb, column_idx_list=column_idx_list,
        user_num=df_X.shape[0],
    )
        
    return data


def load_data(args, params, device):
    uci_path = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
    df_np = np.loadtxt(uci_path + '/raw_data/{}/data/data.txt'.format(args.data))
    df_y = pd.DataFrame(df_np[:, -1:])
    df_X = pd.DataFrame(df_np[:, :-1])
    data = get_data(df_X, df_y, args, params, device)
    return data


