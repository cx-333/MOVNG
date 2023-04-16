# _*_ coding: utf-8 _*_

"""
@Time : 2023/4/6 10:47
@Author : Xiao Chen
@File : secondstage.py
"""
from utils import *
from parse_args import parse_arguments
from model import OptimizeIteration, FirstStage, SecondStage, NewSecondStage
import torch
import warnings
warnings.filterwarnings('ignore')
opt = parse_arguments()


# def __create_saved_model__(vae1_path, vae2_path, vae3_path):
#     model1 = FirstStage(opt.VAE1, opt.NN)
#     model1.load_state_dict(
#         torch.load(vae1_path)
#     )
#     model2 = FirstStage(opt.VAE2, opt.NN)
#     model2.load_state_dict(
#         torch.load(vae2_path)
#     )
#     model3 = FirstStage(opt.VAE3, opt.NN)
#     model3.load_state_dict(
#         torch.load(vae3_path)
#     )
#     return model1, model2, model3


def __get_mean_g__(model, data_list):
    mean, log_var, recon_x, y_pred = model(data_list, infer=True)
    y_indices = prob2class(y_pred)
    adj = get_adj(y_indices)
    g = get_GTCN_g(adj, opt.device)
    return mean, g


def __train_epoch__(first_model, model, dataloader, optimizer):
    ls_value = 0.0
    acc = 0.0
    model.train()
    for data_list, label in dataloader:
        optimizer.zero_grad()
        label = label.to(opt.device)
        mean, g = __get_mean_g__(first_model, data_list)
        loss, y_pred = model(mean, g, label)
        y_indices = prob2class(y_pred)
        acc += accuracy(y_indices, label)
        ls_value += loss.item()
        loss.backward()
        optimizer.step()
    ls_value /= len(dataloader)
    acc /= len(dataloader)
    return ls_value, acc


def __test_epoch__(first_model, model, dataloader):
    acc, w_f1, w_recall = 0.0, 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for data_list, label in dataloader:
            label = label.to(opt.device)
            mean, g = __get_mean_g__(first_model, data_list)
            y_pred = model.infer(mean, g)
            y_indices = prob2class(y_pred)
            acc += accuracy(y_indices, label)
            w_f1 += compute_f1(y_indices, label)
            w_recall += recall(y_indices, label)
    acc /= len(dataloader)
    w_f1 /= len(dataloader)
    w_recall /= len(dataloader)
    return acc, w_f1, w_recall


def __train__(opt):
    early_stop = OptimizeIteration(mode='loss', num=1000)
    first_model = FirstStage(opt.AE1, opt.AE2, opt.AE3, opt.VAE, opt.NN)
    load_checkpoint(first_model, opt.first_stage_save_path)
    model = SecondStage().to(opt.device)
    train_dataloader, test_dataloader = generate_dataloader(opt.data_root, opt.BS, opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)
    if opt.MD == 'train':
        print("Starting Training ... \n")
        for epoch in range(opt.epochs):
            ls, tr_acc = __train_epoch__(first_model, model, train_dataloader, optimizer)
            acc, w_f1, w_recall = __test_epoch__(first_model, model, test_dataloader)
            if epoch % 20 == 0:
                print("Epoch %d : \n" % (epoch + 1))
                print("\tTraining loss: {:.5f}".format(ls))
                print("\tTrain ACC: {:.5f}".format(tr_acc))
                print("\tTest ACC: {:.5f}".format(acc))
                print("\tTest Weight F1: {:.5f}".format(w_f1))
                print("\tTest Weight Recall: {:.5f}".format(w_recall))
            marker = early_stop.detect(ls)
            if marker:
                save_checkpoint(model, opt.second_stage_save_path)
                break
    else:
        print("Starting Testing ...")
        load_checkpoint(model, opt.second_stage_save_path)
        acc, w_f1, w_recall = __test_epoch__(first_model, model, test_dataloader)
        print("\tTest ACC: {:.5f}".format(acc))
        print("\tTest Weight F1: {:.5f}".format(w_f1))
        print("\tTest Weight Recall: {:.5f}".format(w_recall))


def __train_gtcn__(opt):
    train_dataloader, test_dataloader = generate_dataloader(opt.data_root, opt.BS, opt.device)
    model = NewSecondStage(opt).to(opt.device)
    if opt.MD == "test":
        model.test(test_dataloader, opt)
    else:
        model.train_model(train_dataloader, test_dataloader, opt)


"""
Second Stage 
train GTCN

BRCA:   if test the model, changed opt.MD = 'test'. 
        opt.first_stage_save_path = 'checkpoints/first_stage_brca.pt'
        opt.second_stage_save_path = 'checkpoints/second_stage_brca.pt'
        opt.data_root = r'C:\cx\paper\2022002\code\BRCA'
        opt.MD = 'train'
        opt.class_type = 'multiple'
        opt.AE1 = [1000, 512, 128]
        opt.AE2 = [1000, 512, 128]
        opt.AE3 = [503, 256, 64]
        opt.VAE = [128*2+64, 128, 64]
        opt.NN = [64, 40, 32, 5]
        opt.GN = [64, 32, 5]
ROSMAP: if test the model, changed opt.MD = 'test'. 
        opt.first_stage_save_path = 'checkpoints/first_stage_rosmap.pt'
        opt.second_stage_save_path = 'checkpoints/second_stage_rosmap.pt'
        opt.data_root = r'C:\cx\paper\2022002\code\ROSMAP'
        opt.MD = 'train'
        opt.class_type = 'binary'
        opt.AE1 = [200, 512, 128]
        opt.AE2 = [200, 512, 128]
        opt.AE3 = [200, 256, 64]
        opt.VAE = [128*2+64, 128, 64]
        opt.NN = [64, 40, 32, 2]
        opt.GN = [64, 32, 2]
"""


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
    opt.device = 'cuda'
    opt.LR = 1e-2
    # opt.VAE1 = [1000, 512, 128]
    # opt.VAE2 = [1000, 512, 128]
    # opt.VAE3 = [503, 256, 128]
    # # opt.VAE = [128 * 2 + 64, 128, 64]
    # opt.NN = [128*3, 40, 32, 5]
    opt.VAE1 = [200, 256, 128]
    opt.VAE2 = [200, 256, 128]
    opt.VAE3 = [200, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128 * 3, 40, 32, 2]
    opt.GN = [128 * 3, 32, 2]
    # __train__(opt)
    __train_gtcn__(opt)


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
    opt.device = 'cuda'
    opt.LR = 1e-3
    # opt.VAE1 = [1000, 512, 128]
    # opt.VAE2 = [1000, 512, 128]
    # opt.VAE3 = [503, 256, 128]
    # # opt.VAE = [128 * 2 + 64, 128, 64]
    # opt.NN = [128*3, 40, 32, 5]
    opt.VAE1 = [1000, 512, 128]
    opt.VAE2 = [1000, 512, 128]
    opt.VAE3 = [503, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128 * 3, 64, 32, 5]
    opt.GN = [128 * 3, 128, 5]
    # __train__(opt)
    __train_gtcn__(opt)


if __name__ == "__main__":
    opt.MD = 'train'
    if opt.MD == 'test':
        print('Preparing test ROSMAP dataset.')
        run_rosmap(opt, seed=4, mode=opt.MD)
        print('Preparing test BRCA dataset.')
        run_brca(opt, seed=4, mode=opt.MD)
    elif opt.MD == 'train':
        # run_rosmap(opt, seed=4, mode=opt.MD)
        run_brca(opt, seed=6, mode=opt.MD)
