# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/30 21:11
@Author : Xiao Chen
@File : parse_args.py
"""

import argparse
import torch

"""
BRCA: 1000, 1000, 503
ROSMAP: 200, 200, 200
"""


def parse_arguments():
    args = argparse.ArgumentParser(description="Setting of model parameters and \
                                                Dataset.")
    args.add_argument('--AE1', type=int, default=[392_799, 128, 64], nargs='+',
                      help='Dimension setting of AE1.')
    args.add_argument('--AE2', type=int, default=[18_574, 128, 64], nargs='+',
                      help='Dimension setting of AE2.')
    args.add_argument('--AE3', type=int, default=[217, 128, 64], nargs='+',
                      help='Dimension setting of AE3.')
    args.add_argument('--VAE', type=int, default=[64 * 3, 128, 64], nargs='+',
                      help='Dimension setting of VAE.')
    args.add_argument('--VAE1', type=int, default=[392_799, 128, 64], nargs='+',
                      help='The dimensions setting of VAE1 which processes the first '
                           'heterogeneity data and the length of it must be 4.')
    args.add_argument('--VAE2', type=int, default=[18_574, 128, 64], nargs='+',
                      help='The dimensions setting of VAE2 which processes the second '
                           'heterogeneity data and the length of it must be 4.')
    args.add_argument('--VAE3', type=int, default=[217, 128, 64], nargs='+',
                      help='The dimensions setting of VAE3 which processes the third '
                           'heterogeneity data and the length of it must be 4.')
    args.add_argument('--NN', type=int, default=[64, 100, 40, 33], nargs='+',
                      help='The dimensions setting of NN.')
    args.add_argument('--VAE1_save_path', type=str, default=None, help='The save path '
                                                                       'of VAE1.')
    args.add_argument('--VAE2_save_path', type=str, default=None, help='The save path '
                                                                       'of VAE2.')
    args.add_argument('--VAE3_save_path', type=str, default=None, help='The save path '
                                                                       'of VAE3.')
    args.add_argument('--NN_save_path', type=str, default=None, help='The save path '
                                                                     'of NN')
    args.add_argument('--GN', type=int, default=[64, 32, 33], nargs='+',
                      help='The dimensions setting of Graph Networks and'
                           ' the length of it must be 3.')
    args.add_argument('--GN_save_path', type=str, default=None, help='The save path '
                                                                     'of GN')
    args.add_argument('--Hops', type=int, default=10, help='The setting of \
                       hops for GTCN.')
    args.add_argument('--LR', type=float, default=1e-3, help='The setting \
                       of learning rate.')
    args.add_argument('--epochs', type=int, default=1_000_000, help='The number of iteration.')
    args.add_argument('--BS', type=int, default=6, help='The setting of \
                       batch size.')
    args.add_argument('--DS', type=str, default='my_dataset',
                      choices=['my_dataset', 'BRCA', 'ROSMAP'], help='The \
                      selection of dataset. The dimensions setting should be \
                      changed with different dataset.')
    args.add_argument('--data_root', type=str, default=None, help='The path of \
                       dataset.')
    args.add_argument('--first_stage_save_path', type=str, default='checkpoints/first_stage.pt',
                      help='The save path of first stage model.')
    args.add_argument('--second_stage_save_path', type=str, default='checkpoints/second_stage.pt',
                      help='The save path of second stage model.')
    args.add_argument('--finetune_save_path', type=str, default='checkpoints/second_stage.pt',
                      help='The save path of finetune model.')
    args.add_argument('--save_path', type=str, default='checkpoints/movng.pt',
                      help='The save path of MOVNG model.')
    args.add_argument('--class_type', type=str, default='multiple',
                      choices=['multiple', 'binary'],
                      help='The specific classification task.')
    args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                      help='The running environment of code.')
    args.add_argument('--EarlyStop', type=bool, default=True, help='Set True to \
                       avoid over fitting.')
    args.add_argument('--MD', type=str, default='train',
                      choices=['train', 'test'], help='The setting of mode.')
    args.add_argument('--alpha', type=float, default=1e-1, help='The cost of \
                       loss for VAE, which is between 0 and 1.')
    args.add_argument('--beta', type=float, default=1e-1, help='The cost of \
                       loss for NN, which is between 0 and 1.')
    args.add_argument('--gamma', type=float, default=1e-1, help='The cost of \
                       loss for Graph Networks, which is between 0 and 1.')
    opt = args.parse_args()
    opt.device = torch.device(opt.device)
    return opt
