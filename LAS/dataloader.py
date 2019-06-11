import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from tqdm import tqdm


def check_sys_path():
    cwd = os.getcwd()
    if "C:\\Users" in cwd:  # local env
        return "C:/Users/Jeffy/Downloads/Data/hw4/"
    else:  # aws env
        return ""


idx2chr = ["O",
           "Y",
           "X",
           "'",
           "V",
           "G",
           "Z",
           "W",
           "S",
           "M",
           " ",
           "+",
           "A",
           "D",
           "E",
           ".",
           "N",
           "Q",
           "H",
           "T",
           "B",
           "_",
           "P",
           "R",
           "-",
           "L",
           "K",
           "U",
           "F",
           "J",
           ">",
           "I",
           "C",
           "<"]

chr2idx = {chr: i for i, chr in enumerate(idx2chr)}


class TrainData(Dataset):
    def __init__(self, dataset):
        if dataset == "train":
            x_path = "train.npy"
            y_path = "train_transcripts_idx.npy"
        else:
            x_path = "dev.npy"
            y_path = "dev_transcripts_idx.npy"

        self.utter = np.load(os.path.join(check_sys_path(), x_path), encoding="bytes")
        self.transcript = np.load(os.path.join(check_sys_path(), y_path))

    def __getitem__(self, idx):
        return self.utter[idx], self.transcript[idx]

    def __len__(self):
        return self.utter.shape[0]


class TestData(Dataset):
    def __init__(self):
        self.x = np.load(os.path.join(check_sys_path(), "test.npy"), encoding="bytes")
        # self.x = pad_sequence([torch.from_numpy(x) for x in self.x])

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return self.x.shape[0]


def preprocess():
    train_path = "train_transcripts.npy"
    val_path = "dev_transcripts.npy"

    train_transcript = np.load(os.path.join(check_sys_path(), train_path))
    val_transcript = np.load(os.path.join(check_sys_path(), val_path))
    transcript = np.concatenate((train_transcript, val_transcript))

    # "<" means <sos>, ">" means <eos>
    transcript = ["<" + " ".join(s.astype("str")) + ">" for s in transcript]
    idx2chr = list(set([chr for s in transcript for chr in s]))
    chr2idx = {chr: i for i, chr in enumerate(idx2chr)}

    with open(os.path.join(check_sys_path(), "idx2chr.txt"), "w") as f:
        f.writelines("\n".join(idx2chr))

    with open(os.path.join(check_sys_path(), "chr2idx.txt"), "w") as f:
        f.writelines("\n".join(["%s:%d" % (k, i) for k, i in chr2idx.items()]))

    train_transcript = ["<" + " ".join(s.astype("str")) + ">" for s in train_transcript]
    train_transcript = np.array([np.array(list(map(chr2idx.get, s))) for s in train_transcript])
    val_transcript = ["<" + " ".join(s.astype("str")) + ">" for s in val_transcript]
    val_transcript = np.array([np.array(list(map(chr2idx.get, s))) for s in val_transcript])

    np.save(os.path.join(check_sys_path(), "train_transcripts_idx.npy"), train_transcript)
    np.save(os.path.join(check_sys_path(), "dev_transcripts_idx.npy"), val_transcript)


if __name__ == '__main__':
    preprocess()
