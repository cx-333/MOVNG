# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/31 13:46
@Author : Xiao Chen
@File : utils.py
"""

import os
import torch
from torch import Tensor, nn
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from math import sqrt
from dataset import PublicDataset
from torch.utils.data import DataLoader


def get_GTCN_g(A: Tensor, device) -> dict:
    A = A.to_sparse()
    A = clean_A(A)
    g = {}
    A1, A2 = separate(A, norm_type=1)
    A1 = A1.to(device)
    A2 = A2.to(device)
    g['edge_index'] = A1._indices()
    g['edge_weight1'], g['edge_weight2'] = A1._values(), A2
    return g


def clean_A(A):
    s, t = A._indices().tolist()
    N = A.size(0)
    idx = []
    for i in range(len(s)):
        if s[i] == t[i]:
            idx.append(i)
    for i in idx[::-1]:
        del s[i]
        del t[i]
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    return A


def get_adj(x):
    size = len(x)
    adj = torch.eye(size, size)
    i = 0
    while i < size:
        j = i
        while j < size:
            if x[i] == x[j]:
                adj[i, j] = 1
                adj[j, i] = 1
            j += 1
        i += 1
    return adj


def get_adj1(x: Tensor):
    if type(x) != Tensor:
        raise TypeError('Except tensor, but got {0}'.format(type(x)))
    x_len = len(x)
    token = []
    i = 0
    while len(token) < x_len:
        if int(sqrt(i)) ** 2 != i:
            token.append(i)
        i += 1
    token = torch.tensor(token)
    xO = torch.mul(x.reshape(1, -1), torch.sqrt(token))
    xT = xO.T
    temp = torch.matmul(xT, xO)
    b = torch.sqrt(temp)
    result = torch.empty(temp.shape)
    result[b % xO != 0.0] = 0
    result[b % xO == 0.0] = 1
    return result


def separate(A, norm_type=1):
    if norm_type == 1:
        A = norm_adj(A, self_loop=True)
    else:
        A = norm_adj2(A, self_loop=True)
    s, t = A._indices().tolist()
    N = A.size(0)
    values = A._values().tolist()
    value1 = [0] * N
    value2 = []
    s1, t1 = [], []
    for i in range(len(s)):
        if s[i] == t[i]:
            value1[s[i]] = values[i]
        else:
            s1.append(s[i])
            t1.append(t[i])
            value2.append(values[i])
    A1 = torch.sparse_coo_tensor([s1, t1], torch.tensor(value2, dtype=torch.float32), (N, N))
    A2 = torch.tensor(value1, dtype=torch.float32).unsqueeze(-1)
    return A1, A2


# D^-0.5 x A x D^-0.5
def norm_adj(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = torch.pow(degrees, -0.5)
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, torch.sparse.mm(A, D))


# D^-1 x A
def norm_adj2(A, self_loop=True):
    # A is sparse matrix
    s, t = A._indices().tolist()
    N = A.size(0)
    if self_loop:
        s += list(range(N))
        t += list(range(N))
    A = torch.sparse_coo_tensor([s, t], torch.ones(len(s)), (N, N))
    degrees = torch.sparse.sum(A, dim=1).to_dense()
    degrees = 1/degrees
    degrees[torch.isinf(degrees)] = 0
    D = torch.sparse_coo_tensor([list(range(N)), list(range(N))], degrees, (N, N))
    return torch.sparse.mm(D, A)


def multi_class_ls(y_pre, target):
    return nn.functional.cross_entropy(torch.squeeze(y_pre), target.long(), reduction='mean')


def vae_ls(mean, log_var, recon_x, x):
    recon_ls = nn.functional.binary_cross_entropy(recon_x, x)
    kl = - 0.5 * torch.sum(log_var - log_var.exp() - mean.pow(2) + 1)
    return recon_ls + kl


def proposed_vae_ls(mean, log_var, recon_x, x):
    recon_ls = 0
    for rex, trx in zip(recon_x, x):
        recon_ls += nn.functional.binary_cross_entropy(rex, trx)
    kl = - 0.5 * torch.sum(log_var - log_var.exp() - mean.pow(2) + 1)
    return recon_ls + kl


def movng_ls(mean, log_var, recon_x, x,
             y_pre1, y_pre2, target,
             alpha=1, beta=1, gamma=1):
    ls1 = vae_ls(mean, log_var, recon_x, x)
    ls2 = multi_class_ls(y_pre1, target)
    ls3 = multi_class_ls(y_pre2, target)
    return alpha * ls1 + beta * ls2 + gamma * ls3


def prob2class(y_prob):
    return torch.argmax(y_prob, dim=1)


def is_sklearn_processed(x):
    return False if type(x) != np.ndarray or list else True


def accuracy(y_pre, y_true, normalize: bool = True):
    if not is_sklearn_processed(y_pre) or \
       not is_sklearn_processed(y_true):
        y_pre = np.array(y_pre.cpu())
        y_true = np.array(y_true.cpu())
    return accuracy_score(y_true, y_pre, normalize=normalize)


def compute_f1(y_pre, y_true, average: str = 'weighted'):
    if not is_sklearn_processed(y_pre) or \
       not is_sklearn_processed(y_true):
        y_pre = np.array(y_pre.cpu())
        y_true = np.array(y_true.cpu())
    return f1_score(y_true, y_pre, average=average)


def recall(y_pre, y_true, average: str = 'weighted'):
    if not is_sklearn_processed(y_pre) or \
       not is_sklearn_processed(y_true):
        y_pre = np.array(y_pre.cpu())
        y_true = np.array(y_true.cpu())
    return recall_score(y_true, y_pre, average=average)


# Used for mainly BRCA and ROSMAP
def prepare_trte_data(data_folder, device):
    num_view = 3
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view + 1):
        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))

    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        data_tensor_list[i] = data_tensor_list[i].to(device)
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def single_sparse_ls(mean, log_var, recon_x, x, y_pred, label, class_type='multiple'):
    ls1 = vae_ls(mean, log_var, recon_x, x)
    if class_type == 'multiple':
        ls2 = multi_class_ls(y_pred, label)
    else:
        ls2 = binary_class_ls(y_pred, label)
    return ls1 + ls2


def first_stage_ls(mean, log_var, recon_x, x, y_pred, label, class_type='multiple'):
    ls1 = proposed_vae_ls(mean, log_var, recon_x, x)
    if class_type == 'multiple':
        ls2 = multi_class_ls(y_pred, label)
    else:
        ls2 = binary_class_ls(y_pred, label)
    return ls1 + ls2


def second_stage_ls(y_pred, label, class_type='multiple'):
    if class_type == 'multiple':
        return multi_class_ls(y_pred, label)
    else:
        return binary_class_ls(y_pred, label)


def binary_class_ls(y_pred, y_true):
    y_indices = prob2class(y_pred)
    y_indices = torch.tensor(y_indices, dtype=torch.float)
    y_true = torch.tensor(y_true, dtype=torch.float)
    return torch.nn.functional.binary_cross_entropy(y_indices, y_true)


def save_checkpoint(model, checkpoint_path):
    # os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path)
    torch.save(model.state_dict(), filename)
    # print(f"{filename} saved !!!")


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def generate_dataloader(dataset_folder, batch_size, device):
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(dataset_folder, device)
    train_data = PublicDataset(data_tr_list, labels_trte[trte_idx["tr"]])
    test_data = PublicDataset(data_test_list, labels_trte[trte_idx["te"]])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    pass
