# _*_ coding: utf-8 _*_

"""
@Time : 2023/4/6 14:59
@Author : Xiao Chen
@File : finetune.py
"""
import numpy as np

from model import FineTune, opt, OptimizeIteration
from utils import *
import torch
import warnings
warnings.filterwarnings('ignore')


def __train_epoch__(loss_fn, model, optimizer, train_dataloader):
    ls, tr_acc = 0.0, 0.0
    model.train()
    for data, label in train_dataloader:
        label = label.to(opt.device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = loss_fn(y_pred, label, opt.class_type)
        y_indices = prob2class(y_pred)
        loss.backward()
        optimizer.step()
        ls += loss.item()
        tr_acc += accuracy(y_indices, label)
    ls /= len(train_dataloader)
    tr_acc /= len(train_dataloader)
    # print("Epoch %d:\n Train Loss: %.5f" % (epoch + 1, ls))
    return ls, tr_acc


def __test_epoch__(model, test_dataloader):
    acc, w_f1, w_recall = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for data_list, label in test_dataloader:
            y_pred = model(data_list)
            y_indices = prob2class(y_pred)
            acc += accuracy(y_indices, label)
            w_f1 += compute_f1(y_indices, label)
            w_recall += recall(y_indices, label)
    acc /= len(test_dataloader)
    w_f1 /= len(test_dataloader)
    w_recall /= len(test_dataloader)
    return acc, w_f1, w_recall


def __train__(opt):
    expect = 0.0
    early_stop = OptimizeIteration(mode='loss', num=20)
    model = FineTune(opt).to(opt.device)
    loss_fn = second_stage_ls
    train_dataloader, test_dataloader = generate_dataloader(opt.data_root, opt.BS, opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)
    if opt.MD == 'train':
        print("Starting Training ... \n")
        for epoch in range(opt.epochs):
            ls, tr_acc = __train_epoch__(loss_fn, model, optimizer, train_dataloader)
            acc, w_f1, w_recall = __test_epoch__(model, test_dataloader)
            # if epoch % 20 == 0:
            print("Epoch %d : \n" % (epoch + 1))
            print("\tTraining loss: {:.5f}".format(ls))
            print("\tTrain ACC: {:.5f}".format(tr_acc))
            print("\tTest ACC: {:.5f}".format(acc))
            print("\tTest Weight F1: {:.5f}".format(w_f1))
            print("\tTest Weight Recall: {:.5f}".format(w_recall))
            marker = early_stop.detect(ls)
            if acc > expect:
                expect = acc
                save_checkpoint(model, opt.finetune_save_path)
            if marker:
                break
        # save_checkpoint(model, opt.finetune_save_path)
    else:
        print("Starting Testing ...")
        load_checkpoint(model, opt.finetune_save_path)
        acc, w_f1, w_recall = __test_epoch__(model, test_dataloader)
        print("\tTest ACC: {:.5f}".format(acc))
        print("\tTest Weight F1: {:.5f}".format(w_f1))
        print("\tTest Weight Recall: {:.5f}".format(w_recall))


def run_rosmap(opt, seed=4, mode='train'):
    set_seed(seed)
    opt.VAE1_save_path = 'checkpoints/vae1_rosmap.pt'
    opt.VAE2_save_path = 'checkpoints/vae2_rosmap.pt'
    opt.VAE3_save_path = 'checkpoints/vae3_rosmap.pt'
    opt.NN_save_path = 'checkpoints/neuralnet_rosmap.pt'
    opt.data_root = r'C:\cx\paper\2022002\code\ROSMAP'
    opt.MD = mode
    opt.class_type = 'multiple'
    opt.GN_save_path = 'checkpoints/gtcn_rosmap.pt'
    opt.finetune_save_path = 'checkpoints/finetune_rosmap.pt'
    opt.device = 'cuda'
    opt.LR = 1e-4
    opt.VAE1 = [200, 256, 128]
    opt.VAE2 = [200, 256, 128]
    opt.VAE3 = [200, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128*3, 40, 32, 2]
    opt.GN = [128*3, 32, 2]
    __train__(opt)


def run_brca(opt, seed=4, mode='train'):
    set_seed(seed)
    opt.VAE1_save_path = 'checkpoints/vae1_brca.pt'
    opt.VAE2_save_path = 'checkpoints/vae2_brca.pt'
    opt.VAE3_save_path = 'checkpoints/vae3_brca.pt'
    opt.NN_save_path = 'checkpoints/neuralnet_brca.pt'
    opt.data_root = r'C:\cx\paper\2022002\code\BRCA'
    opt.MD = mode
    opt.class_type = 'multiple'
    opt.GN_save_path = 'checkpoints/gtcn_brca.pt'
    opt.finetune_save_path = 'checkpoints/finetune_brca.pt'
    opt.device = 'cuda'
    opt.LR = 1e-5
    opt.VAE1 = [1000, 512, 128]
    opt.VAE2 = [1000, 512, 128]
    opt.VAE3 = [503, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128 * 3, 64, 32, 5]
    opt.GN = [128 * 3, 128, 5]
    __train__(opt)


if __name__ == "__main__":
    opt.MD = 'test'
    if opt.MD == 'test':
        print('Preparing test ROSMAP dataset.')
        run_rosmap(opt, seed=0, mode=opt.MD)
        print('Preparing test BRCA dataset.')
        run_brca(opt, seed=109, mode=opt.MD)
    elif opt.MD == 'train':
        # run_rosmap(opt, seed=0, mode=opt.MD)
        run_brca(opt, seed=109, mode=opt.MD)

