import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from PIL import Image
from tqdm import tqdm


class TrainData(Dataset):
    def __init__(self, dataset):
        if dataset == "train":
            x_path = "wsj0_train.npy"
            y_path = "wsj0_train_merged_labels.npy"
        else:
            x_path = "wsj0_dev.npy"
            y_path = "wsj0_dev_merged_labels.npy"

        self.x = np.load(os.path.join(check_sys_path(), x_path), encoding="latin1")
        self.y = np.load(os.path.join(check_sys_path(), y_path), encoding="latin1")

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.y.shape[0]


class TestData(Dataset):
    def __init__(self):
        self.x = np.load(os.path.join(check_sys_path(), "transformed_test_data.npy"), encoding="bytes")
        self.x = pad_sequence([torch.from_numpy(x) for x in self.x])

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


def check_sys_path():
    cwd = os.getcwd()
    if "C:\\Users" in cwd:  # local env
        return "C:/Users/Jeffy/Downloads/Data/hw3"
    else:  # aws env
        return ""


if __name__ == '__main__':
    self = TestData()

    train_loader = DataLoader(self,
                              batch_size=128,
                              num_workers=2,
                              shuffle=True)
    loader = iter(train_loader)
    X, y = loader.next()
    X, y = loader.next()


