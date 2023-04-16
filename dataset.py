# _*_ coding: utf-8 _*_

"""
@Time : 2023/3/31 16:19
@Author : Xiao Chen
@File : dataset.py
"""

from torch.utils.data import Dataset


class PublicDataset(Dataset):
    def __init__(self, data_list, data_idx):
        super(PublicDataset, self).__init__()
        self.data_list = data_list
        self.data_idx = data_idx

    def __getitem__(self, index):
        result = [data[index] for data in self.data_list]
        label = self.data_idx[index]
        return result, label

    def __len__(self):
        return len(self.data_idx)

