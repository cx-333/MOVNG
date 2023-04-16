# _*_ coding: utf-8 _*_

"""
@Time : 2023/4/4 20:55
@Author : Xiao Chen
@File : firststage.py
"""

from model import FirstStage, OptimizeIteration, NeuralNetwork, NewFirstStage
from parse_args import parse_arguments
from utils import *
import warnings
warnings.filterwarnings('ignore')
opt = parse_arguments()


def __train_epoch__(model, dataloader, optimizer):
    ls_value = 0.0
    model.train()
    for data_list, label in dataloader:
        optimizer.zero_grad()
        label = label.to(opt.device)
        loss, mean, log_var, recon_x, y_pred = model(data_list, label)
        ls_value += loss.item()
        loss.backward()
        optimizer.step()
    ls_value /= len(dataloader)
    return ls_value


def __test_epoch__(model, dataloader):
    acc = 0.0
    w_f1 = 0.0
    w_recall = 0.0
    model.eval()
    with torch.no_grad():
        for data_list, label in dataloader:
            label = label.to(opt.device)
            mean, log_var, recon_x, y_pred = model.infer(data_list)
            y_pred = prob2class(y_pred)
            acc += accuracy(y_pred, label)
            w_f1 += compute_f1(y_pred, label)
            w_recall += recall(y_pred, label)
    acc /= len(dataloader)
    w_f1 /= len(dataloader)
    w_recall /= len(dataloader)
    return acc, w_f1, w_recall


def __train__(opt):
    early_stop = OptimizeIteration(mode='loss', num=50)
    model = FirstStage(opt.AE1, opt.AE2, opt.AE3, opt.VAE, opt.NN).to(opt.device)
    train_dataloader, test_dataloader = generate_dataloader(opt.data_root, opt.BS, opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)
    if opt.MD == 'train':
        print("Starting Training ... \n")
        for epoch in range(opt.epochs):
            ls = __train_epoch__(model, train_dataloader, optimizer)
            acc, w_f1, w_recall = __test_epoch__(model, test_dataloader)
            if epoch % 20 == 0:
                print("Epoch %d : \n" % (epoch + 1))
                print("\tTraining loss: {:.5f}".format(ls))
                print("\tTest ACC: {:.5f}".format(acc))
                print("\tTest Weight F1: {:.5f}".format(w_f1))
                print("\tTest Weight Recall: {:.5f}".format(w_recall))
            marker = early_stop.detect(ls)
            if marker:
                save_checkpoint(model, opt.first_stage_save_path)
                break
    else:
        print("Starting Testing ...")
        load_checkpoint(model, opt.first_stage_save_path)
        acc, w_f1, w_recall = __test_epoch__(model, test_dataloader)
        print("\tTest ACC: {:.5f}".format(acc))
        print("\tTest Weight F1: {:.5f}".format(w_f1))
        print("\tTest Weight Recall: {:.5f}".format(w_recall))


def __train_nnet__(opt):
    train_dataloader, test_dataloader = generate_dataloader(opt.data_root, opt.BS, opt.device)
    model = NewFirstStage(opt).to(opt.device)
    if opt.MD == "test":
        model.test(test_dataloader, opt)
    elif opt.MD == 'train':
        model.train_model(train_dataloader, test_dataloader, opt)
        opt.NN[0] = opt.VAE1[-1] + opt.VAE2[-1] + opt.VAE3[-1]
        model.train_nnet(train_dataloader, test_dataloader, opt)


"""
First Stage 
train sparse fusion representation

BRCA:   if test the model, changed opt.MD = 'test'. 
        opt.first_stage_save_path = 'checkpoints/first_stage_brca.pt'
        opt.data_root = r'C:\cx\paper\2022002\code\BRCA'
        opt.MD = 'train'
        opt.class_type = 'multiple'
        # opt.AE1 = [1000, 512, 128]
        # opt.AE2 = [1000, 512, 128]
        # opt.AE3 = [503, 256, 64]
        # opt.VAE = [128*2+64, 128, 64]
        # opt.NN = [64, 40, 32, 5]
        
        opt.VAE1 = [1000, 512, 128]
        opt.VAE2 = [1000, 512, 128]
        opt.VAE3 = [503, 256, 128]
        # opt.VAE = [128 * 2 + 64, 128, 64]
        opt.NN = [128, 40, 32, 5]
ROSMAP: if test the model, changed opt.MD = 'test'. 
        opt.first_stage_save_path = 'checkpoints/first_stage_rosmap.pt'
        opt.data_root = r'C:\cx\paper\2022002\code\ROSMAP'
        opt.MD = 'train'
        opt.class_type = 'binary'
        opt.AE1 = [200, 512, 128]
        opt.AE2 = [200, 512, 128]
        opt.AE3 = [200, 256, 64]
        opt.VAE = [128*2+64, 128, 64]
        opt.NN = [64, 40, 32, 2]

"""


def run_brca(opt, seed=4, mode="train"):
    set_seed(seed)
    # opt.first_stage_save_path = 'checkpoints/first_stage_brca.pt'
    opt.VAE1_save_path = 'checkpoints/vae1_brca.pt'
    opt.VAE2_save_path = 'checkpoints/vae2_brca.pt'
    opt.VAE3_save_path = 'checkpoints/vae3_brca.pt'
    opt.NN_save_path = 'checkpoints/neuralnet_brca.pt'
    opt.data_root = r'C:\cx\paper\2022002\code\BRCA'
    opt.MD = mode
    opt.class_type = 'multiple'
    opt.VAE1 = [1000, 512, 128]
    opt.VAE2 = [1000, 512, 128]
    opt.VAE3 = [503, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128, 64, 32, 5]
    opt.LR = 1e-3
    opt.device = 'cuda'
    # __train__(opt)
    __train_nnet__(opt)


def run_rosmap(opt, seed=4, mode="train"):
    set_seed(seed)
    # opt.first_stage_save_path = 'checkpoints/first_stage_brca.pt'
    opt.VAE1_save_path = 'checkpoints/vae1_rosmap.pt'
    opt.VAE2_save_path = 'checkpoints/vae2_rosmap.pt'
    opt.VAE3_save_path = 'checkpoints/vae3_rosmap.pt'
    opt.NN_save_path = 'checkpoints/neuralnet_rosmap.pt'
    opt.data_root = r'C:\cx\paper\2022002\code\ROSMAP'
    opt.MD = mode
    opt.class_type = 'multiple'
    opt.VAE1 = [200, 256, 128]
    opt.VAE2 = [200, 256, 128]
    opt.VAE3 = [200, 256, 128]
    # opt.VAE = [128 * 2 + 64, 128, 64]
    opt.NN = [128, 40, 32, 2]
    opt.LR = 1e-3
    opt.device = 'cuda'
    # __train__(opt)
    __train_nnet__(opt)


if __name__ == "__main__":
    opt.MD = 'train'
    if opt.MD == 'test':
        print('Preparing test ROSMAP dataset.')
        run_rosmap(opt, seed=4, mode=opt.MD)
        print('Preparing test BRCA dataset.')
        run_brca(opt, seed=4, mode=opt.MD)
    elif opt.MD == 'train':
        # run_rosmap(opt, seed=4, mode=opt.MD)
        run_brca(opt, seed=201, mode=opt.MD)
