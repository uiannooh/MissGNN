import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score
from sklearn.metrics import roc_auc_score

from models.gnn_model import get_gnn
from models.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, mask_edge

def train_gnn_y(data, args, params, load_path, ckpt_path, device = torch.device('cpu')):
    x = data.x.clone().detach().to(device)
    y = data.y.clone().detach().to(device)

    model = get_gnn(data, args, params).to(device)

    if params['impute_hiddens'] == '': impute_hiddens = []
    else: impute_hiddens = list(map(int, params['impute_hiddens'].split('_')))

    if args.concat_states: input_dim = params['node_dim'] * len(model.convs) * 2
    else: input_dim = params['node_dim'] * 2

    impute_model = MLPNet(
        input_dim, 1,
        hidden_layer_sizes = impute_hiddens,
        hidden_activation = args.impute_activation,
        dropout = args.dropout
    ).to(device)

    if params['predict_hiddens'] == '': predict_hiddens = []
    else: predict_hiddens = list(map(int, params['predict_hiddens'].split('_')))

    row, col = data.df_X.shape
    if args.task == 'reg': output_dim = 1
    else : output_dim = len(y.unique())

    predict_model = MLPNet(
        col, output_dim,
        hidden_layer_sizes = predict_hiddens,
        dropout = args.dropout
    ).to(device)

    trainable_parameters = list(model.parameters()) \
                           + list(impute_model.parameters()) \
                           + list(predict_model.parameters())

    scheduler, opt = build_optimizer(args, params, trainable_parameters)

    if args.pre_train == True:
        model.load_state_dict(torch.load(load_path + 'mdi_model.pt'))

    if args.task == 'class':
        cross_entropy = torch.nn.CrossEntropyLoss()
        softmax = torch.nn.Softmax(dim = 1)
        num_classes = len(y.unique())
        accuracy = Accuracy(task = 'multiclass', num_classes = num_classes).to(device)
        f1_score = F1Score(task = 'multiclass', num_classes = num_classes, average = 'weighted').to(device)
        print('num_classes:', num_classes)

    x = data.x.clone().detach().to(device)
    column_idx_list = data.column_idx_list
    edge_index_all = data.edge_index_all.clone().detach().to(device)
    train_edge_index = data.train_edge_index
    train_edge_attr = data.train_edge_attr
    all_train_y_mask = data.train_y_mask.clone().detach().to(device)
    test_y_mask = data.test_y_mask.clone().detach().to(device)
    if args.valid > 0.:
        torch.manual_seed(args.seed)
        valid_mask = get_known_mask(args.valid, all_train_y_mask.shape[0]).to(device)
        valid_mask = valid_mask * all_train_y_mask
        train_y_mask = all_train_y_mask.clone().detach()
        train_y_mask[valid_mask] = False
        valid_y_mask = all_train_y_mask.clone().detach()
        valid_y_mask[~valid_mask] = False
    else:
        train_y_mask = all_train_y_mask.clone().detach()

    for epoch in range(params['epochs_y']):
        model.train()
        impute_model.train()
        predict_model.train()

        opt.zero_grad()

        x_embd_sub = []
        for i in range(len(column_idx_list)):
            known_mask = get_known_mask(args.known, int(train_edge_attr[i].shape[0]/2)).to(device)
            double_known_mask = torch.cat([known_mask, known_mask])
            known_edge_index, known_edge_attr = mask_edge(train_edge_index[i], train_edge_attr[i], double_known_mask, True)

            x_embd_sub.append(model(x, known_edge_attr, known_edge_index))

        x_embd = torch.mean(torch.stack(x_embd_sub), axis = 0)
        X = impute_model([x_embd[edge_index_all[0, :int(row * col)]], x_embd[edge_index_all[1, :int(row * col)]]])
        X = torch.reshape(X, [row, col])
        if args.task == 'reg':
            pred = predict_model(X)[:, 0]
        else:
            pred = predict_model(X)
        pred_train = pred[train_y_mask]
        label_train = y[train_y_mask]

        if args.task == 'reg':
            loss = F.mse_loss(pred_train, label_train)
        else:
            label_train = label_train.to(torch.int64)
            loss = cross_entropy(pred_train, label_train)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        if scheduler is not None:
            scheduler.step(epoch)

        model.eval()
        impute_model.eval()
        predict_model.eval()
        
        with torch.no_grad():
            if args.valid > 0.:
                x_embd_sub = []
                for i in range(len(column_idx_list)):
                    x_embd_sub.append(model(x, train_edge_attr[i], train_edge_index[i]))
                x_embd = torch.mean(torch.stack(x_embd_sub), axis = 0)
                X = impute_model([x_embd[edge_index_all[0, :int(row * col)]], x_embd[edge_index_all[1, :int(row * col)]]])
                X = torch.reshape(X, [row, col])
                if args.task == 'reg':
                    pred = predict_model(X)[:, 0]
                else:
                    pred = predict_model(X)
                pred_valid = pred[valid_y_mask]
                label_valid = y[valid_y_mask]
                if args.task == 'reg':
                    mse = F.mse_loss(pred_valid, label_valid)
                    valid_rmse = np.sqrt(mse.item())
                    l1 = F.l1_loss(pred_valid, label_valid)
                    valid_l1 = l1.item()
                else:
                    label_valid = label_valid.to(torch.int64)
                    class_valid = torch.argmax(pred_valid, axis = 1)
                    # softmax_valid = softmax(pred_valid)
                    valid_acc = accuracy(class_valid, label_valid).item()
                    # if num_classes > 2:
                    #     valid_auc = roc_auc_score(label_valid.cpu(), softmax_valid.cpu(), multi_class = "ovo").item()
                    # else:
                    #     valid_auc = roc_auc_score(label_valid.cpu(), softmax_valid[:, 1].cpu()).item()
                    valid_ce_loss = cross_entropy(pred_valid, label_valid).item()
                
            x_embd_sub = []
            for i in range(len(column_idx_list)):
                x_embd_sub.append(model(x, train_edge_attr[i], train_edge_index[i]))
            x_embd = torch.mean(torch.stack(x_embd_sub), axis = 0)
            X = impute_model([x_embd[edge_index_all[0, :int(row * col)]], x_embd[edge_index_all[1, :int(row * col)]]])
            X = torch.reshape(X, [row, col])
            if args.task == 'reg':
                pred = predict_model(X)[:, 0]
            else:
                pred = predict_model(X)
            pred_test = pred[test_y_mask]
            label_test = y[test_y_mask]
            if args.task == 'reg':
                mse = F.mse_loss(pred_test, label_test)
                test_rmse = np.sqrt(mse.item())
                l1 = F.l1_loss(pred_test, label_test)
                test_l1 = l1.item()
            else:
                label_test = label_test.to(torch.int64)
                class_test = torch.argmax(pred_test, axis = 1)
                softmax_test = softmax(pred_test)
                test_acc = accuracy(class_test, label_test).item()
                test_f1 = f1_score(class_test, label_test).item()
                if num_classes > 2:
                    test_auc = roc_auc_score(label_test.cpu(), softmax_test.cpu(), multi_class = "ovo").item()
                else:
                    test_auc = roc_auc_score(label_test.cpu(), softmax_test[:, 1].cpu()).item()
            
            if epoch % 100 == 0:
                print('epoch:', epoch)
                print('loss:', train_loss)
                if args.task == 'reg':
                    if args.valid > 0.:
                        print('valid rmse:', valid_rmse)
                        print('valid l1:', valid_l1)
                    print('test rmse:', test_rmse)
                    print('test l1:', test_l1)
                else:
                    if args.valid > 0.:
                        print('valid loss:', valid_ce_loss)
                        print('valid accuracy:', valid_acc)
                        # print("valid auc:", valid_auc)
                    print('test accuracy:', test_acc)
                    print('test auc:', test_auc)
                print()
    
    print('seed', args.seed)
    print('----------test result----------')
    if args.task == 'reg':
        print('test mse:', mse.item())
        print('test rmse:', test_rmse)
        print('test l1:', test_l1)
    else:
        print('test accuracy:', test_acc)
        print('test f1:', test_f1)
        print('test auc:', test_auc)
    print()    

    torch.save(model.state_dict(), ckpt_path + 'y_model.pt')

    if args.task == 'reg': return valid_l1 if args.valid > 0. else train_loss
    else: return valid_ce_loss if args.valid > 0. else train_loss 