# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/30 21:11
@Author : Xiao Chen
@File : parse_args.py
"""

import argparse


def parse_arguments():
    args = argparse.ArgumentParser(description="Setting of model parameters and \
                                                Dataset.")
    args.add_argument('-A1', '--AE1', type=list, default=[392_799, 512, 256],
                      help='The dimensions setting of the first omics data in AE1.')
    args.add_argument('-A2', '--AE2', type=list, default=[18_574, 256, 128],
                      help='The dimensions setting of the second omics data in AE2.')
    args.add_argument('-A3', '--AE3', type=list, default=[217, 128, 64],
                      help='The dimensions setting of the third omics data in AE3.')
    args.add_argument('-V', '--VAE', type=list, default=[256 + 128 + 64, 128, 64],
                      help='The dimensions setting of VAE.')
    args.add_argument('-N', '--NN', type=list, default=[64, 32, 33],
                      help='The dimensions setting of NN.')
    args.add_argument('-G', '--GN', type=list, default=[64, 32, 33],
                      help='The dimensions setting of Graph Networks.')
    args.add_argument('-H', '--Hops', type=int, default=10, help='The setting of \
                       hops for GTCN.')
    args.add_argument('-L', '--LR', type=float, default=1e-3, help='The setting \
                       of learning rate.')
    args.add_argument('-B', '--BS', type=int, default=32, help='The setting of \
                       batch size.')
    args.add_argument('-D', '--DS', type=str, default='my_dataset',
                      choices=['my_dataset', 'BRCA', 'ROSMAP'], help='The \
                      selection of dataset. The dimensions setting should be \
                      changed with different dataset.')
    args.add_argument('--data_root', type=str, default=None, help='The path of \
                       dataset.')
    args.add_argument('-M', '--MD', type=str, default='test',
                      choices=['train', 'test'], help='The setting of mode.')
    args.add_argument('--alpha', type=float, default=1e-1, help='The cost of \
                       loss for VAE, which is between 0 and 1.')
    args.add_argument('--beta', type=float, default=1e-1, help='The cost of \
                       loss for NN, which is between 0 and 1.')
    args.add_argument('--gamma', type=float, default=1e-1, help='The cost of \
                       loss for Graph Networks, which is between 0 and 1.')
    opt = args.parse_args()
    return opt
