# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/30 21:03
@Author : Xiao Chen
@File : model.py
"""
import numpy as np
import torch
from torch import nn
from parse_args import parse_arguments
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_scatter import scatter
from utils import *

opt = parse_arguments()


class AutoEncoder(nn.Module):
    def __init__(self, dims: list, p: float = 0.5) -> None:
        super().__init__()
        dims_len = len(dims)
        if dims_len < 2:
            raise ValueError('The length of dims at least 2, but got {0}.'.format(dims_len))
        encode = []
        for i in range(0, dims_len - 1):
            encode.append(nn.Linear(dims[i], dims[i + 1]))
            encode.append(nn.BatchNorm1d(dims[i + 1]))
            encode.append(nn.ReLU())
            encode.append(nn.Dropout(p))
        self.encode = nn.Sequential(*encode)

        decode = []
        for i in range(dims_len - 1, 1, -1):
            decode.append(nn.Linear(dims[i], dims[i - 1]))
            decode.append(nn.BatchNorm1d(dims[i - 1]))
            decode.append(nn.ReLU())
            decode.append(nn.Dropout(p))
        decode.append(nn.Linear(dims[1], dims[0]))
        decode.append(nn.BatchNorm1d(dims[0]))
        decode.append(nn.Sigmoid())
        self.decode = nn.Sequential(*decode)

    def forward(self, x, mode: str = 'encode'):
        if mode == 'encode':
            out = self.encode(x)
        elif mode == 'decode':
            out = self.decode(x)
        else:
            raise ValueError('The mode either is encode or decode, but got {0}'.format(mode))
        return out


class VAE(nn.Module):
    def __init__(self, dims: list, dropout_p: float = 0.5) -> None:
        super(VAE, self).__init__()
        hidden_len = len(dims)
        # Encoder
        if hidden_len - 1 <= 1:
            raise f'The length of hidden_dim at least is 2, but get {hidden_len - 1}.'
        # elif hidden_len == 1:
        #     self.encoder = self.fc_layer(input_dim, hidden_dim[0])
        else:
            self.encoder = nn.Sequential(
                nn.Linear(dims[0], dims[1]),
                nn.BatchNorm1d(dims[1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_p))
            for i in range(2, hidden_len - 1):
                self.encoder.add_module(f'e_{i + 1}_l', nn.Linear(dims[i - 1], dims[i]))
                self.encoder.add_module(f'e_{i + 1}_b', nn.BatchNorm1d(dims[i]))
                self.encoder.add_module(f'e_{i + 1}_r', nn.ReLU())
                self.encoder.add_module(f'e_{i + 1}_d', nn.Dropout(p=dropout_p))
            self.e_mean = nn.Sequential(
                nn.Linear(dims[-2], dims[-1]),
                nn.BatchNorm1d(dims[-1]))
            self.e_log_var = nn.Sequential(
                nn.Linear(dims[-2], dims[-1]),
                nn.BatchNorm1d(dims[-1]))
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(dims[-1], dims[-2]),
            nn.BatchNorm1d(dims[-2]),
            nn.ReLU(),
            nn.Dropout(p=dropout_p))
        for i in range(2, hidden_len - 1):
            self.decoder.add_module(f'd_{i}_l', nn.Linear(dims[-i], dims[-i - 1]))
            self.decoder.add_module(f'd_{i}_b', nn.BatchNorm1d(dims[-i - 1]))
            self.decoder.add_module(f'd_{i}_r', nn.ReLU())
            self.decoder.add_module(f'd_{i}_d', nn.Dropout(p=dropout_p))
        self.decoder.add_module('d_last_l', nn.Linear(dims[1], dims[0]))
        self.decoder.add_module('d_last_b', nn.BatchNorm1d(dims[0]))
        self.decoder.add_module('d_last_s', nn.Sigmoid())

    def encode(self, x):
        temp = self.encoder(x)
        mean = self.e_mean(temp)
        log_var = self.e_log_var(temp)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z)
        return mean, log_var, z, recon_x


class NeuralNetwork(nn.Module):
    def __init__(self, dims: list):
        super().__init__()
        dims_len = len(dims)
        if dims_len != 4:
            raise ValueError('The length of dims must equal to 4, but got {0}.'.format(dims_len))
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ProposedVAE(nn.Module):
    def __init__(self,
                 AE1_dims,
                 AE2_dims,
                 AE3_dims,
                 VAE_dims):
        super().__init__()
        VAE_dims_len = len(VAE_dims)
        self.AE1_dims = AE1_dims
        self.AE2_dims = AE2_dims
        if VAE_dims_len <= 2:
            raise ValueError("The length of dims in VAE at least 3.")
        self.ae1 = AutoEncoder(AE1_dims)
        self.ae2 = AutoEncoder(AE2_dims)
        self.ae3 = AutoEncoder(AE3_dims)
        encoder, decoder = [], []
        for i in range(0, VAE_dims_len - 2):
            encoder.extend([
                nn.Linear(VAE_dims[i], VAE_dims[i + 1]),
                nn.BatchNorm1d(VAE_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            ])
        self.encoder = nn.Sequential(*encoder)
        self.mean = nn.Sequential(
            nn.Linear(VAE_dims[-2], VAE_dims[-1]),
            nn.BatchNorm1d(VAE_dims[-1])
        )
        self.log_var = nn.Sequential(
            nn.Linear(VAE_dims[-2], VAE_dims[-1]),
            nn.BatchNorm1d(VAE_dims[-1])
        )
        for i in range(VAE_dims_len - 1, 0, -1):
            decoder.extend([
                nn.Linear(VAE_dims[i], VAE_dims[i - 1]),
                nn.BatchNorm1d(VAE_dims[i - 1]),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            ])
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x: list):
        temp1 = self.ae1(x[0])
        temp2 = self.ae2(x[1])
        temp3 = self.ae3(x[2])
        temp = torch.cat([temp1, temp2, temp3], dim=1)
        mean, log_var = self.encode(temp)
        z = self.reparameterization(mean, log_var)
        recon_x = self.decode(z)
        [z1, z2, z3] = torch.tensor_split(recon_x,
                                          [self.AE1_dims[-1],
                                           self.AE1_dims[-1] +
                                           self.AE2_dims[-1]],
                                          dim=1)
        recon_x1 = self.ae1(z1, mode='decode')
        recon_x2 = self.ae2(z2, mode='decode')
        recon_x3 = self.ae3(z3, mode='decode')
        return mean, log_var, recon_x1, recon_x2, recon_x3

    def encode(self, x):
        temp = self.encoder(x)
        mean = self.mean(temp)
        log_var = self.log_var(temp)
        return mean, log_var

    def reparameterization(self, mean, log_var):
        sigma = torch.exp(0.5 * log_var)
        eps = torch.randn_like(sigma)
        return mean + eps * sigma

    def decode(self, z):
        recon_x = self.decoder(z)
        return recon_x


class SingleSparse(nn.Module):
    def __init__(self,
                 VAE_dims,
                 NN_dims):
        super().__init__()
        self.vae = VAE(VAE_dims)
        self.neuralnet = NeuralNetwork(NN_dims)

    def forward(self, x, label=None, infer=False):
        mean, log_var, z, recon_x = self.vae(x)
        y_pred = self.neuralnet(mean)
        if infer:
            return mean, log_var, recon_x, y_pred
        loss = single_sparse_ls(mean, log_var, recon_x, x, y_pred, label, opt.class_type)
        return loss, mean, log_var, recon_x, y_pred

    def infer(self, x):
        return self.forward(x, infer=True)


class OptimizeIteration:
    def __init__(self, mode: str = 'loss', num: int = 20):
        # avoid over fitting by loss or accuracy.
        self.expect = 0
        self.mode = mode
        self.decide_early_stop = []
        self.num = num

    def detect(self, data):
        self.decide_early_stop.append(data)
        length = len(self.decide_early_stop)
        if length > self.num:
            back = self.decide_early_stop[-(self.num - 1):]
            front = self.decide_early_stop[-self.num:-1]
            decide_var = np.array(front) - np.array(back)
            if self.mode == 'loss':
                if sum(decide_var >= 0) < sum(decide_var < 0):
                    return True
                else:
                    return False
            elif self.mode == 'accuracy':
                if sum(decide_var >= 0) > sum(decide_var < 0):
                    return True
                else:
                    return False


# First stage: Training sparse fusion representation model
class FirstStage(nn.Module):
    def __init__(self,
                 AE1_dims,
                 AE2_dims,
                 AE3_dims,
                 VAE_dims,
                 NN_dims
                 ):
        super().__init__()
        self.provae = ProposedVAE(AE1_dims, AE2_dims, AE3_dims, VAE_dims)
        self.neuralnet = NeuralNetwork(NN_dims)

    def forward(self, x: list, label=None, infer=False):
        mean, log_var, recon_x1, recon_x2, recon_x3 = self.provae(x)
        y_pred = self.neuralnet(mean)
        recon_x = [recon_x1, recon_x2, recon_x3]
        if infer:
            return mean, log_var, recon_x, y_pred
        loss = first_stage_ls(mean, log_var, recon_x, x, y_pred, label, opt.class_type)
        return loss, mean, log_var, recon_x, y_pred

    def infer(self, x: list):
        return self.forward(x, infer=True)


class NewFirstStage(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.vae1 = VAE(opt.VAE1)
        self.vae2 = VAE(opt.VAE2)
        self.vae3 = VAE(opt.VAE3)
        self.nnet = NeuralNetwork(opt.NN)
        self.action = {
            0: self.vae1,
            1: self.vae2,
            2: self.vae3
        }

    def forward(self, x: list, mode: int):
        action = self.action.get(mode)
        mean, log_var, z, recon_x = action(x[mode])
        out = self.nnet(mean)
        return mean, log_var, recon_x, out

    def train_model(self, train_dataloader, test_dataloader, opt):
        modes = [0, 1, 2]
        save_paths = [opt.VAE1_save_path, opt.VAE2_save_path, opt.VAE3_save_path]
        loss_fn = single_sparse_ls
        optimizer = torch.optim.Adam(self.parameters(), opt.LR)
        for mode in modes:
            expect = 0
            print("Starting Training : The %d -th model ... " % (mode + 1))
            earlystop = OptimizeIteration(mode='loss', num=20)
            for epoch in range(opt.epochs):
                self.train()
                ls_value, tr_acc = 0.0, 0.0
                for data_list, label in train_dataloader:
                    label = label.to(opt.device)
                    optimizer.zero_grad()
                    mean, log_var, recon_x, out = self.forward(data_list, mode)
                    loss = loss_fn(mean, log_var, recon_x, data_list[mode], out, label, class_type=opt.class_type)
                    loss.backward()
                    optimizer.step()
                    ls_value += loss.item()
                    out_indices = prob2class(out)
                    tr_acc += accuracy(out_indices, label)
                ls_value /= len(train_dataloader)
                tr_acc /= len(train_dataloader)
                with torch.no_grad():
                    self.eval()
                    acc = 0.0
                    w_f1 = 0.0
                    w_recall = 0.0
                    for data_list, label in test_dataloader:
                        label = label.to(opt.device)
                        mean, log_var, recon_x, out = self.forward(data_list, mode)
                        out_indices = prob2class(out)
                        acc += accuracy(out_indices, label)
                        w_f1 += compute_f1(out_indices, label)
                        w_recall += recall(out_indices, label)
                    acc /= len(test_dataloader)
                    w_f1 /= len(test_dataloader)
                    w_recall /= len(test_dataloader)
                # if (epoch + 1) % 20 == 0:
                print("Epoch %d :" % (epoch + 1))
                print("\t\tThe %d model train loss: %.6f\t train acc: %6f " % (mode + 1, ls_value, tr_acc))
                print("\t\tThe %d model test acc: %.6f\t test weighted f1_score: %6f"
                      "\t test weighted recall: %.6f " % (mode+1, acc, w_f1, w_recall))
                stop = earlystop.detect(ls_value)
                if acc > expect:
                    expect = acc
                    save_checkpoint(self.action[mode], save_paths[mode])
                if stop:
                    # torch.save(self.state_dict(), save_paths[mode])
                    break
            # save_checkpoint(self.action[mode], save_paths[mode])
        print("Done !!! ")

    def load_model(self, opt):
        model = NewFirstStage(opt)
        model.action[0].load_state_dict(
            torch.load(opt.VAE1_save_path), strict=False
        )
        model.action[1].load_state_dict(
            torch.load(opt.VAE2_save_path), strict=False
        )
        model.action[2].load_state_dict(
            torch.load(opt.VAE3_save_path), strict=False
        )
        return model.action[0], model.action[1], model.action[2]

    def train_nnet(self, train_dataloader, test_dataloader, opt):
        expect = 0
        model1, model2, model3 = self.load_model(opt)
        model1, model2, model3 = model1.to(opt.device), model2.to(opt.device), model3.to(opt.device)
        self.nnet = NeuralNetwork(opt.NN).to(opt.device)
        optimizer = torch.optim.Adam(self.nnet.parameters(), opt.LR)
        earlystop = OptimizeIteration(mode='loss', num=20)
        loss_fn = second_stage_ls
        print("Starting training Neural Network ...")
        for epoch in range(opt.epochs):
            self.nnet.train()
            ls_value, tr_acc = 0.0, 0.0
            for data_list, label in train_dataloader:
                label = label.to(opt.device)
                optimizer.zero_grad()
                mean1, log_var1, z1, recon_x1 = model1(data_list[0])
                mean2, log_var2, z2, recon_x2 = model2(data_list[1])
                mean3, log_var3, z3, recon_x3 = model3(data_list[2])
                mean = torch.cat([mean1, mean2, mean3], dim=1)
                y_pred = self.nnet(mean)
                y_indices = prob2class(y_pred)
                loss = loss_fn(y_pred, label, opt.class_type)
                loss.backward()
                optimizer.step()
                ls_value += loss.item()
                tr_acc += accuracy(y_indices, label)
            ls_value /= len(train_dataloader)
            tr_acc /= len(train_dataloader)
            with torch.no_grad():
                self.nnet.eval()
                acc = 0.0
                w_f1 = 0.0
                w_recall = 0.0
                for data_list, label in test_dataloader:
                    label = label.to(opt.device)
                    mean1, log_var1, z1, recon_x1 = model1(data_list[0])
                    mean2, log_var2, z2, recon_x2 = model2(data_list[1])
                    mean3, log_var3, z3, recon_x3 = model3(data_list[2])
                    mean = torch.cat([mean1, mean2, mean3], dim=1)
                    y_pred = self.nnet(mean)
                    y_indices = prob2class(y_pred)
                    acc += accuracy(y_indices, label)
                    w_f1 += compute_f1(y_indices, label)
                    w_recall += recall(y_indices, label)
                acc /= len(test_dataloader)
                w_f1 /= len(test_dataloader)
                w_recall /= len(test_dataloader)
            # if (epoch + 1) % 20 == 0:
            print("Epoch %d :" % (epoch+1))
            print("\t\tNeural network model train loss: %.6f\t train acc: %6f " % (ls_value, tr_acc))
            print("\t\tNeural network model test acc: %.6f\t test weighted f1_score: %6f"
                  "\t test weighted recall: %.6f " % (acc, w_f1, w_recall))
            stop = earlystop.detect(ls_value)
            if acc > expect:
                expect = acc
                save_checkpoint(self.nnet, opt.NN_save_path)
            if stop:
                break
        # save_checkpoint(self.nnet, opt.NN_save_path)
        print("Done !!! ")

    def test(self, test_dataloader, opt):
        model1, model2, model3 = self.load_model(opt)
        model1, model2, model3 = model1.to(opt.device), model2.to(opt.device), model3.to(opt.device)
        opt.NN[0] = opt.VAE1[-1] + opt.VAE2[-1] + opt.VAE3[-1]
        self.nnet = NeuralNetwork(opt.NN).to(opt.device)
        load_checkpoint(self.nnet, opt.NN_save_path)
        model1.eval()
        model2.eval()
        model3.eval()
        self.nnet.eval()
        print("Testing ... ")
        with torch.no_grad():
            acc = 0.0
            w_f1 = 0.0
            w_recall = 0.0
            for data_list, label in test_dataloader:
                label = label.to(opt.device)
                mean1, log_var1, z1, recon_x1 = model1(data_list[0])
                mean2, log_var2, z2, recon_x2 = model2(data_list[1])
                mean3, log_var3, z3, recon_x3 = model3(data_list[2])
                mean = torch.cat([mean1, mean2, mean3], dim=1)
                y_pred = self.nnet(mean)
                y_indices = prob2class(y_pred)
                acc += accuracy(y_indices, label)
                w_f1 += compute_f1(y_indices, label)
                w_recall += recall(y_indices, label)
        acc /= len(test_dataloader)
        w_f1 /= len(test_dataloader)
        w_recall /= len(test_dataloader)
        print("\t\tFirst stage test acc: %.6f\t weighted f1_score: %6f"
              "\t weighted recall: %.6f " % (acc, w_f1, w_recall))


class NewSecondStage(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.newfstage = NewFirstStage(opt)
        self.newfstage.action[0].load_state_dict(
            torch.load(opt.VAE1_save_path), strict=False
        )
        self.newfstage.action[0].to(opt.device)

        self.newfstage.action[1].load_state_dict(
            torch.load(opt.VAE2_save_path), strict=False
        )
        self.newfstage.action[1].to(opt.device)

        self.newfstage.action[2].load_state_dict(
            torch.load(opt.VAE3_save_path), strict=False
        )
        self.newfstage.action[2].to(opt.device)

        self.newfstage.nnet.load_state_dict(
            torch.load(opt.NN_save_path), strict=False
        )
        self.newfstage.nnet.to(opt.device)

        self.gtcn = GTCN(opt.GN).to(opt.device)
        self.device = opt.device

    def forward(self, x: list):
        mean1, log_var1, z1, recon_x1 = self.newfstage.action[0](x[0])
        mean2, log_var2, z2, recon_x2 = self.newfstage.action[1](x[1])
        mean3, log_var3, z3, recon_x3 = self.newfstage.action[2](x[2])
        mean = torch.cat([mean1, mean2, mean3], dim=1)
        y_pred1 = self.newfstage.nnet(mean)
        y_pre_indices = prob2class(y_pred1)
        adj = get_adj(y_pre_indices)
        g = get_GTCN_g(adj, self.device)
        y_pred = self.gtcn(mean, g)
        return y_pred

    def train_model(self, train_dataloader, test_dataloader, opt):
        expect = 0
        optimizer = torch.optim.Adam(self.gtcn.parameters(), opt.LR)
        earlystop = OptimizeIteration(mode='loss', num=20)
        loss_fn = second_stage_ls
        print("Starting training Graph Tree Convolution Network ...")
        for epoch in range(opt.epochs):
            self.gtcn.train()
            ls_value, tr_acc = 0.0, 0.0
            for data_list, label in train_dataloader:
                label = label.to(opt.device)
                optimizer.zero_grad()
                y_pred = self.forward(data_list)
                y_indices = prob2class(y_pred)
                loss = loss_fn(y_pred, label, opt.class_type)
                loss.backward()
                optimizer.step()
                ls_value += loss.item()
                tr_acc += accuracy(y_indices, label)
            ls_value /= len(train_dataloader)
            tr_acc /= len(train_dataloader)
            with torch.no_grad():
                self.gtcn.eval()
                acc = 0.0
                w_f1 = 0.0
                w_recall = 0.0
                for data_list, label in test_dataloader:
                    label = label.to(opt.device)
                    y_pred = self.forward(data_list)
                    y_indices = prob2class(y_pred)
                    acc += accuracy(y_indices, label)
                    w_f1 += compute_f1(y_indices, label)
                    w_recall += recall(y_indices, label)
                acc /= len(test_dataloader)
                w_f1 /= len(test_dataloader)
                w_recall /= len(test_dataloader)
            # if (epoch + 1) % 20 == 0:
            print("Epoch %d :" % (epoch + 1))
            print("\t\tGTCN model train loss: %.6f\t train acc: %6f " % (ls_value, tr_acc))
            print("\t\tGTCN model test acc: %.6f\t test weighted f1_score: %6f"
                  "\t test weighted recall: %.6f " % (acc, w_f1, w_recall))
            stop = earlystop.detect(ls_value)
            if acc > expect:
                expect = acc
                save_checkpoint(self.gtcn, opt.GN_save_path)
            if stop:
                break
        # save_checkpoint(self.gtcn, opt.GN_save_path)
        print("Done !!! ")

    def test(self, test_dataloader, opt):
        load_checkpoint(self.gtcn, opt.GN_save_path)
        self.gtcn.eval()
        print("Testing ... ")
        with torch.no_grad():
            acc = 0.0
            w_f1 = 0.0
            w_recall = 0.0
            for data_list, label in test_dataloader:
                label = label.to(opt.device)
                y_pred = self.forward(data_list)
                y_indices = prob2class(y_pred)
                acc += accuracy(y_indices, label)
                w_f1 += compute_f1(y_indices, label)
                w_recall += recall(y_indices, label)
        acc /= len(test_dataloader)
        w_f1 /= len(test_dataloader)
        w_recall /= len(test_dataloader)
        print("\t\tSecond stage test acc: %.6f\t weighted f1_score: %6f"
              "\t weighted recall: %.6f " % (acc, w_f1, w_recall))


class FineTune(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.newsstage = NewSecondStage(opt)
        load_checkpoint(self.newsstage.gtcn, opt.GN_save_path)

    def forward(self, x: list):
        y_pred = self.newsstage(x)
        return y_pred

# class GTCN(MessagePassing):
#     def __init__(self, dims: list, p: float = 0.5, hop: int = 10):
#         super().__init__(aggr='add')
#         dims_len = len(dims)
#         if dims_len < 2:
#             raise ValueError('The length of dims at least 2, but got {0}.'.format(dims_len))
#
#         self.hops = hop
#         layer1 = []
#         for i in range(0, dims_len - 2):
#             layer1.append(nn.Dropout(p))
#             layer1.append(nn.Linear(dims[i], dims[i + 1]))
#             layer1.append(nn.ReLU())
#             layer1.append((nn.Dropout(p)))
#         self.layer1 = nn.Sequential(*layer1)
#         self.dropout = nn.Dropout(p)
#
#         layer2 = [nn.Linear(dims[-2], dims[-1]),
#                   nn.Softmax(dim=1)]
#         self.layer2 = nn.Sequential(*layer2)
#
#     def forward(self, x, g):
#         edge_index = g['edge_index']
#         edge_weight1, edge_weight2 = g['edge_weight1'], g['edge_weight2']
#         x = self.layer1(x)
#         h = x
#         for k in range(0, self.hops):
#             h = self.propagate(edge_index, x=h, edge_weight=edge_weight1)
#             h = h + edge_weight2 * x
#             h = self.dropout(h)
#         h = self.layer2(h)
#         return h
#
#     def message(self, x_j, edge_weight):
#         return edge_weight.view(-1, 1) * x_j
class GTCN(MessagePassing):
    def __init__(self, GN_dims, dropout=0.5, dropout2=0.5, hop=10):
        super().__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout2)
        self.hop = hop
        self.relu = nn.ReLU()
        self.layer1 = nn.Linear(GN_dims[0], GN_dims[1])
        self.layer2 = nn.Linear(GN_dims[1], GN_dims[2])
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.layer1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.layer2.weight, gain=1.414)

    def forward(self, x, g):
        edge_index = g['edge_index']
        edge_weight1, edge_weight2 = g['edge_weight1'], g['edge_weight2']
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        h = x
        for k in range(self.hop):
            h = self.propagate(edge_index, x=h, edge_weight=edge_weight1)
            h = h + edge_weight2 * x
            h = self.dropout2(h)
        h = self.layer2(h)
        h = torch.nn.functional.softmax(h, dim=1)
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
        for i in range(len(n_dims) - 1):
            self.layers.append(GCN_layer(n_dims[i], n_dims[i + 1]))

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
                h = h / div
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
                h = h / div
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
    def __init__(self):
        super().__init__()
        self.first_stage = FirstStage(opt.AE1, opt.AE2, opt.AE3, opt.VAE, opt.NN)
        load_checkpoint(self.first_stage, opt.first_stage_save_path)
        self.second_stage = SecondStage()
        load_checkpoint(self.second_stage, opt.second_stage_save_path)

    def forward(self, x: list, label: Tensor = None, infer: bool = False):
        if infer:
            mean, log_var, recon_x, y_pred1 = self.first_stage.infer(x)
            y_indices = prob2class(y_pred1)
            adj = get_adj(y_indices)
            g = get_GTCN_g(adj, opt.device)
            y_pred2 = self.second_stage.infer(mean, g)
            return y_pred2
        else:
            loss1, mean, log_var, recon_x, y_pred1 = self.first_stage(x, label)
            y_indices = prob2class(y_pred1)
            adj = get_adj(y_indices)
            g = get_GTCN_g(adj, opt.device)
            loss2, y_pred2 = self.second_stage(mean, g, label)
            loss = loss1 + loss2
            return loss, y_pred2

    def infer(self, data_list):
        y_pred = self.forward(data_list, infer=True)
        return y_pred


class SparseRepresentation(nn.Module):
    def __init__(self,
                 AE1_dims,
                 AE2_dims,
                 AE3_dims,
                 VAE_dims,
                 NN_dims):
        super().__init__()
        self.VAE = VAE(VAE_dims, AE1_dims, AE2_dims, AE3_dims)
        self.NN = NeuralNetwork(NN_dims)

    def forward(self, x: list, label: Tensor = None, infer: bool = False):
        mean, log_var, z, recon_x = self.VAE(x)
        y_pred = self.NN(mean)
        if infer:
            return mean, log_var, recon_x, y_pred
        loss = single_sparse_ls(mean, log_var, recon_x, x, y_pred, label)
        return mean, log_var, recon_x, loss, y_pred

    def infer(self, x):
        y_pred = self.forward(x, infer=True)
        return y_pred


class SecondStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.gtcn = GTCN(opt.GN)

    def forward(self, mean, g, label: Tensor = None, infer: bool = False):
        y_pred = self.gtcn(mean, g)
        if infer:
            return y_pred
        loss = second_stage_ls(y_pred, label, opt.class_type)
        return loss, y_pred

    def infer(self, mean, g):
        y_pred = self.forward(mean, g, infer=True)
        return y_pred


def sparse_representation():
    sparse_representation = SparseRepresentation(opt.AE1,
                                                 opt.AE2,
                                                 opt.AE3,
                                                 opt.VAE,
                                                 opt.NN)
    sparse_representation.load_state_dict(
        torch.load(opt.sparse_representation_path),
        strict=False
    )
    return sparse_representation


if __name__ == '__main__':
    device = torch.device('cuda')
    data = [torch.rand(10, 32_124), torch.rand(10, 18_574), torch.rand(10, 217)]

    vae = ProposedVAE([32_124, 256, 128, 64], [18_574, 128, 64],
                      [217, 128, 64], [64 * 3, 128, 64])
    print(vae)
    print(vae(data))
