import torch
import numpy as np

def subsets_generator(df, args, params):
    n_subsets = params['n_subsets']
    n_columns = df.shape[1]
    overlap = params['overlap']
    n_columns_subset = int(n_columns / n_subsets)
    n_overlap = int(overlap * n_columns_subset)

    column_idx = torch.tensor(range(n_columns))
    permuted_order = np.random.permutation(n_subsets) if args.mode == 'train' else range(n_subsets)
    column_idx_list = []

    print('n_subsets:', n_subsets)
    print('overlap:', overlap)
    # print('n_column:', n_columns)
    # print('n_overlap:', n_overlap)

    if n_subsets != 1:
        for i in permuted_order:
            if i == 0:
                start_idx = 0
                stop_idx = n_columns_subset + n_overlap
            else:
                start_idx = i * n_columns_subset - n_overlap
                stop_idx = (i + 1) * n_columns_subset
            column_idx_list.append(column_idx[start_idx:stop_idx])
        if stop_idx != n_columns:
            start_idx = n_columns - (n_columns_subset + n_overlap)
            stop_idx = n_columns
            column_idx_list.append(column_idx[start_idx:stop_idx])  # the last subset might thus be used again due to the randomness of permuted_order in training mode
    else:
        column_idx_list.append(column_idx[:])

    if len(column_idx_list) == 1:
        column_idx_list.append(column_idx_list[0])
        
    # print('column_idx_list:', column_idx_list)

    return column_idx_list