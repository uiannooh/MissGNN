import numpy as np
import torch
import torch.nn.functional as F

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, mask_edge

def train_gnn_mdi(data, args, params, ckpt_path, device = torch.device('cpu')):
    model = get_gnn(data, args, params).to(device)

    if params['impute_hiddens'] == '': impute_hiddens = []
    else: impute_hiddens = list(map(int, params['impute_hiddens'].split('_')))

    if args.concat_states: input_dim = params['node_dim'] * len(model.convs) * 2
    else: input_dim = params['node_dim'] * 2
    
    output_dim = 1
    impute_model = MLPNet(
        input_dim, output_dim,
        hidden_layer_sizes = impute_hiddens,
        hidden_activation = args.impute_activation,
        dropout = args.dropout
    ).to(device)

    trainable_parameters = list(model.parameters()) + list(impute_model.parameters())
    scheduler, opt = build_optimizer(args, params, trainable_parameters)


    x_comb = data.x_comb.clone().detach().to(device)
    column_idx_comb = data.column_idx_comb
    all_train_edge_index_comb = data.train_edge_index_comb
    all_train_edge_attr_comb = data.train_edge_attr_comb
    all_train_labels_comb = data.train_labels_comb
    test_input_edge_index_comb = all_train_edge_index_comb
    test_input_edge_attr_comb = all_train_edge_attr_comb
    test_edge_index_comb = data.test_edge_index_comb
    # test_edge_attr_comb = data.test_edge_attr_comb
    test_labels_comb = data.test_labels_comb
    if args.valid > 0.:
        valid_edge_index_comb = []
        # valid_edge_attr_comb = []
        valid_labels_comb = []
        train_edge_index_comb = []
        train_edge_attr_comb = []
        train_labels_comb = []
        for i in range(len(column_idx_comb)):    
            valid_mask = get_known_mask(args.valid, int(all_train_edge_attr_comb[i].shape[0]/4))
            double_valid_mask = valid_mask.repeat(2)
            valid_labels_comb.append(all_train_labels_comb[i][double_valid_mask])
            train_labels_comb.append(all_train_labels_comb[i][~double_valid_mask])
            quad_valid_mask = double_valid_mask.repeat(2)
            valid_edge_index, valid_edge_attr = mask_edge(all_train_edge_index_comb[i], all_train_edge_attr_comb[i], quad_valid_mask, True)
            train_edge_index, train_edge_attr = mask_edge(all_train_edge_index_comb[i], all_train_edge_attr_comb[i], ~quad_valid_mask, True)
            valid_edge_index_comb.append(valid_edge_index)
            # valid_edge_attr_comb.append(valid_edge_attr)
            train_edge_index_comb.append(train_edge_index)
            train_edge_attr_comb.append(train_edge_attr)
    else:
        train_edge_index_comb, train_edge_attr_comb, train_labels_comb = \
            all_train_edge_index_comb, all_train_edge_attr_comb, all_train_labels_comb
    
    for epoch in range(params['epochs_mdi']):
        model.train()
        impute_model.train()

        opt.zero_grad()

        total_loss = 0
        for i in range(len(column_idx_comb)):
            known_mask = get_known_mask(args.known, int(train_edge_attr_comb[i].shape[0]/4)).to(device)
            quad_known_mask = known_mask.repeat(4)
            known_edge_index, known_edge_attr = mask_edge(train_edge_index_comb[i], train_edge_attr_comb[i], quad_known_mask, True)
        
            x_embd = model(x_comb, known_edge_attr, known_edge_index)
            pred = impute_model([x_embd[train_edge_index_comb[i][0]], x_embd[train_edge_index_comb[i][1]]])
            a, b = torch.split(pred, int(pred.shape[0]/2))
            a = a[:int(a.shape[0]/2), 0]
            b = b[:int(b.shape[0]/2), 0]
            pred_train = torch.cat([a, b])
            label_train = train_labels_comb[i]
            loss = F.mse_loss(pred_train, label_train)
            total_loss += loss
        total_loss = total_loss / len(column_idx_comb)
        total_loss.backward()
        opt.step()
        train_loss = total_loss.item()
        if scheduler is not None:
            scheduler.step(epoch)

        model.eval()
        impute_model.eval()

        with torch.no_grad():
            if args.valid > 0.:
                valid_rmse = 0
                valid_l1 = 0
                for i in range(len(column_idx_comb)):
                    x_embd = model(x_comb, train_edge_attr_comb[i], train_edge_index_comb[i])
                    pred = impute_model([x_embd[valid_edge_index_comb[i][0], :], x_embd[valid_edge_index_comb[i][1], :]])
                    a, b = torch.split(pred, int(pred.shape[0]/2))
                    a = a[:int(a.shape[0]/2), 0]
                    b = b[:int(b.shape[0]/2), 0]
                    pred_valid = torch.cat([a, b])
                    label_valid = valid_labels_comb[i]
                    mse = F.mse_loss(pred_valid, label_valid)
                    valid_rmse += np.sqrt(mse.item())
                    l1 = F.l1_loss(pred_valid, label_valid)
                    valid_l1 += l1.item()
                valid_rmse = valid_rmse / len(column_idx_comb)
                valid_l1 = valid_l1 / len(column_idx_comb)

            test_rmse = 0
            test_l1 = 0
            for i in range(len(column_idx_comb)):
                x_embd = model(x_comb, test_input_edge_attr_comb[i], test_input_edge_index_comb[i])
                pred = impute_model([x_embd[test_edge_index_comb[i][0], :], x_embd[test_edge_index_comb[i][1], :]])
                a, b = torch.split(pred, int(pred.shape[0]/2))
                a = a[:int(a.shape[0]/2), 0]
                b = b[:int(b.shape[0]/2), 0]
                pred_test = torch.cat([a, b])
                label_test = test_labels_comb[i]
                mse = F.mse_loss(pred_test, label_test)
                test_rmse += np.sqrt(mse.item())
                l1 = F.l1_loss(pred_test, label_test)
                test_l1 += l1.item()
            test_rmse = test_rmse / len(column_idx_comb)
            test_l1 = test_l1 / len(column_idx_comb)
            
            if epoch % 100 == 0:
                print('epoch:', epoch)
                print('loss:', train_loss)
                if args.valid > 0.:
                    print('valid rmse:', valid_rmse)
                    print('valid l1:', valid_l1)
                print('test rmse:', test_rmse)
                print('test l1:', test_l1)
                print()

            torch.save(model.state_dict(), ckpt_path + 'mdi_model.pt')

    return valid_l1 if args.valid > 0. else train_loss