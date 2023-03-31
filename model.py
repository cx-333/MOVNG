# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/30 21:03
@Author : Xiao Chen
@File : model.py
"""
import torch
from torch import nn
from parse_args import parse_arguments
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_scatter import scatter
from utils import *

opt = parse_arguments()


class AutoEncoder(nn.Module):
    def __init__(self, dims: list, p: float = 0.2) -> None:
        super().__init__()
        dims_len = len(dims)
        if dims_len < 2:
            raise ValueError('The length of dims at least 2, but got {0}.'.format(dims_len))
        layers = []
        for i in range(0, dims_len - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        return out


class VAE(nn.Module):
    def __init__(self, dims: list):
        super().__init__()
        dims_len = len(dims)
        if dims_len < 3:
            raise ValueError('The length of dims at least 3, but got {0}.'.format(dims_len))
        # Encoder
        encoder = []
        for i in range(0, dims_len - 2):
            encoder.append(nn.Linear(dims[i], dims[i + 1]))
            encoder.append(nn.BatchNorm1d(dims[i + 1]))
            encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)
        self.mean = nn.Linear(dims[-2], dims[-1])
        self.log_var = nn.Linear(dims[-2], dims[-1])
        # Decoder
        decoder = []
        for i in range(dims_len - 1, 1, -1):
            decoder.append(nn.Linear(dims[i], dims[i - 1]))
            decoder.append(nn.BatchNorm1d(dims[i - 1]))
            decoder.append(nn.ReLU())
        decoder.append(nn.Linear(dims[1], dims[0]))
        decoder.append(nn.BatchNorm1d(dims[0]))
        decoder.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z)
        return mean, log_var, z, recon_x

    def encode(self, x):
        temp = self.encoder(x)
        mean = self.mean(temp)
        log_var = self.log_var(temp)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + sigma * eps

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x


class NeuralNetwork(nn.Module):
    def __init__(self, dims: list, p: float = 0.2):
        super().__init__()
        dims_len = len(dims)
        if dims_len < 2:
            raise ValueError('The length of dims at least 2, but got {0}.'.format(dims_len))

        net = []
        for i in range(0, dims_len - 2):
            net.append(nn.Linear(dims[i], dims[i + 1]))
            net.append(nn.BatchNorm1d(dims[i + 1]))
            net.append(nn.ReLU())
            net.append(nn.Dropout(p))
        net.append(nn.Linear(dims[-2], dims[-1]))
        net.append(nn.Softmax(dim=1))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class GTCN(MessagePassing):
    def __init__(self, dims: list, p: float = 0.5, hop: int = 10):
        super().__init__(aggr='add')
        dims_len = len(dims)
        if dims_len < 2:
            raise ValueError('The length of dims at least 2, but got {0}.'.format(dims_len))

        self.hops = hop
        layer1 = []
        for i in range(0, dims_len - 2):
            layer1.append(nn.Dropout(p))
            layer1.append(nn.Linear(dims[i], dims[i + 1]))
            layer1.append(nn.ReLU())
            layer1.append((nn.Dropout(p)))
        self.layer1 = nn.Sequential(*layer1)
        self.dropout = nn.Dropout(p)

        layer2 = [nn.Linear(dims[-2], dims[-1]),
                  nn.Softmax(dim=1)]
        self.layer2 = nn.Sequential(*layer2)

    def forward(self, x, g):
        edge_index = g['edge_index']
        edge_weight1, edge_weight2 = g['edge_weight1'], g['edge_weight2']
        x = self.layer1(x)
        h = x
        for k in range(0, self.hops):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight1)
            h = h + edge_weight2 * x
            h = self.dropout(h)
        h = self.layer2(h)
        return h

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


class GAT(nn.Module):
    def __init__(self, n_dims, n_heads, dropout=0.6, attn_dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.layers = nn.ModuleList()
        n_in = n_dims[0]
        for i in range(len(n_dims) - 2):
            self.layers.append(
                GATConv(n_in, n_dims[i + 1], n_heads[i], concat=True, negative_slope=alpha, dropout=attn_dropout))
            n_in = n_dims[i + 1] * n_heads[i]
        self.layers.append(
            GATConv(n_in, n_dims[-1], n_heads[-1], concat=False, negative_slope=alpha, dropout=attn_dropout))

    def forward(self, x, g):
        edge_index = g['edge_index']
        x = self.dropout(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = self.elu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class GCN(nn.Module):
    def __init__(self, n_dims, dropout=0.5):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layers = nn.ModuleList()
        for i in range(len(n_dims)-1):
            self.layers.append(GCN_layer(n_dims[i], n_dims[i+1]))

    def forward(self, x, g):
        A = g['A']
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, A)
            x = self.relu(x)
            x = self.dropout(x)
        return self.layers[-1](x, A)


class GCN_layer(nn.Module):
    def __init__(self, n_in, n_out):
        super(GCN_layer, self).__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.reset_param()

    def forward(self, x, A):
        x = self.linear(x)
        return torch.sparse.mm(A, x)

    def reset_param(self):
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)


class GTAN(nn.Module):
    def __init__(self, n_in, n_hid, n_out, dropout=0.5, attn_dropout=0.5,
                 hop=10, layerwise=True, zero_init=False):
        super(GTAN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(attn_dropout)
        self.hop = hop
        self.layerwise = layerwise
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(n_in, n_hid)
        if not self.layerwise:
            self.attn1 = nn.Linear(n_hid, 1, bias=False)
            self.attn2 = nn.Linear(n_hid, 1, bias=False)
        else:
            self.attn1 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
            self.attn2 = nn.ModuleList(nn.Linear(n_hid, 1, bias=False) for _ in range(hop))
        self.fc2 = nn.Linear(n_hid, n_out)
        self.elu = nn.ELU()
        self.reset_parameters(zero_init)

    def forward(self, x, g):
        N = x.size(0)
        s, t = g['edge_index']
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        h = x
        if not self.layerwise:
            for i in range(self.hop):
                x1 = self.attn1(x)
                h1 = self.attn2(h)
                w2 = x1 + self.attn2(x)
                w1 = x1[s] + h1[t]
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
        else:
            for i in range(self.hop):
                x1 = self.attn1[i](x)
                h1 = self.attn2[i](h)
                w1 = x1[s] + h1[t]
                w2 = x1 + self.attn2[i](x)
                w1 = torch.exp(self.leakyrelu(w1))
                w2 = torch.exp(self.leakyrelu(w2))
                div = scatter(w1, s, dim=0, dim_size=N) + w2
                h = scatter(w1 * h[t], s, dim=0, dim_size=N) + w2 * x
                h = h/div
                h = self.elu(h)
                h = self.dropout2(h)
        h = self.fc2(h)
        return h

    def reset_parameters(self, zero_init):
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.414)
        if not self.layerwise:
            if zero_init:
                nn.init.zeros_(self.attn1.weight)
                nn.init.zeros_(self.attn2.weight)
            else:
                nn.init.xavier_uniform_(self.attn1.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_uniform_(self.attn2.weight, gain=nn.init.calculate_gain('relu'))
        else:
            if zero_init:
                for i in range(self.hop):
                    nn.init.zeros_(self.attn1[i].weight)
                    nn.init.zeros_(self.attn2[i].weight)
            else:
                for i in range(self.hop):
                    nn.init.xavier_uniform_(self.attn1[i].weight, gain=nn.init.calculate_gain('relu'))
                    nn.init.xavier_uniform_(self.attn2[i].weight, gain=nn.init.calculate_gain('relu'))


class MOVNG(nn.Module):
    def __init__(self,
                 AE1_dims,
                 AE2_dims,
                 AE3_dims,
                 VAE_dims,
                 NN_dims,
                 GN_dims,
                 device):
        super().__init__()
        self.device = device
        self.AE1 = AutoEncoder(AE1_dims)
        self.AE2 = AutoEncoder(AE2_dims)
        self.AE3 = AutoEncoder(AE3_dims)
        self.VAE = VAE(VAE_dims)
        self.NN = NeuralNetwork(NN_dims)
        self.GN = GTCN(GN_dims)

    def forward(self, x: list, label: Tensor = None, infer: bool = False):
        temp1 = self.AE1(x[0])
        temp2 = self.AE2(x[1])
        temp3 = self.AE3(x[2])
        temp = torch.cat([temp1, temp2, temp3], dim=1)
        mean, log_var, z, recon_x = self.VAE(temp)
        y_pre1 = self.NN(mean)
        y_pre_indices = prob2class(y_pre1)
        adj = get_adj(y_pre_indices)
        g = get_GTCN_g(adj, self.device)
        y_pred = self.GN(mean, g)
        if infer:
            return y_pred
        loss = movng_ls(mean, log_var, recon_x, temp, y_pre1, y_pred, label,
                        opt.alpha, opt.beta, opt.gamma)
        return loss, y_pred

    def infer(self, data_list):
        y_pred = self.forward(data_list, infer=True)
        return y_pred


if __name__ == '__main__':
    device = torch.device('cuda')
    data1 = torch.rand(10, 392_799).to(device)
    data2 = torch.rand(10, 18_574).to(device)
    data3 = torch.rand(10, 217).to(device)
    data = [data1, data2, data3]
    label = torch.ones(10).to(device)
    model = MOVNG(opt.AE1, opt.AE2, opt.AE3, opt.VAE, opt.NN, opt.GN, device).to(device)
    print(model(data, label))
