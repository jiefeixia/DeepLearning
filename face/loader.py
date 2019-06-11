import os
from glob import glob

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data.dataset import Dataset
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


def ClassificationTrain(dataset, large=False):
    if dataset == "train":
        path = os.path.join(check_sys(), "train_data/large" if large else "train_data/medium")
    elif dataset == "validation":
        path = os.path.join(check_sys(), "validation_classification/large" if large else "validation_classification/medium")
    else:
        path = os.path.join(check_sys(), "test_classification")

    print("loading data from", path)
    data_transform = transforms.Compose([transforms.ToTensor()])
    return ImageFolder(path, transform=data_transform)


class ClassificationTest(Dataset):
    def __init__(self):
        path = check_sys() + "/test_classification/medium"
        print("loading data from", path)

        imgs = []
        with open(os.path.join(check_sys(), "test_order_classification.txt")) as f:
            for line in f:
                imgs.append(line.strip())
        data = []
        for img in imgs:
            # change axis from (channel, height, width) to (channel, height, width)
            data.append(np.moveaxis(np.asarray(Image.open(os.path.join(path, img))), -1, 0))
        self.data = torch.FloatTensor(np.array(data))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.data.shape[0]


class VerificationTrain(Dataset):
    def __init__(self, dataset):
        if dataset == "train":
            path = os.path.join(check_sys(), "train_data/medium")
            print("loading data from", path)

            len_id = len(os.listdir(path))
            random_ids = np.random.randint(0, len_id, len_id)
            neg_ids = random_ids + np.random.randint(len_id - 1)
            neg_ids[neg_ids >= len_id] = neg_ids[neg_ids >= len_id] - len_id
            self.paths = [list(np.random.choice(glob("%s/%d/*" % (path, random_ids[i])), 2)) +
                          [np.random.choice(glob("%s/%d/*" % (path, neg_ids[i])))] for i in range(len_id)]

        else:  # dataset == "validation"
            path = os.path.join(check_sys(), "validation_verification")
            print("loading data from", path)

            len_id = len(os.listdir(path))
            random_ids = np.random.randint(0, len_id, len_id)
            neg_ids = random_ids + np.random.randint(len_id - 1)
            neg_ids[neg_ids >= len_id] = neg_ids[neg_ids >= len_id] - len_id
            self.paths = [list(np.random.choice(glob("%s/fid_%d/*" % (path, random_ids[i])), 2)) +
                           [np.random.choice(glob("%s/fid_%d/*" % (path, neg_ids[i])))] for i in range(len_id)]

    def __getitem__(self, item):
        return (torch.FloatTensor(np.moveaxis(np.asarray(Image.open(self.paths[item][0])), -1, 0)),
                torch.FloatTensor(np.moveaxis(np.asarray(Image.open(self.paths[item][1])), -1, 0)),
                torch.FloatTensor(np.moveaxis(np.asarray(Image.open(self.paths[item][2])), -1, 0)))
        # return [self.paths[item][i] for i in range(3)]

    def __len__(self):
        return len(self.paths)


class VerificationTest(Dataset):
    def __init__(self):
        path = os.path.join(check_sys(), "test_verification/")
        print("loading data from", path)

        self.pairs = []
        with open(os.path.join(check_sys(), "test_trials_verification_student.txt")) as f:
            for line in f:
                self.pairs.append([os.path.join(path + img) for img in line.strip().split(" ")])

    def __getitem__(self, item):
        return (torch.FloatTensor(np.moveaxis(np.asarray(Image.open(self.pairs[item][0])), -1, 0)),
                torch.FloatTensor(np.moveaxis(np.asarray(Image.open(self.pairs[item][1])), -1, 0)))
        # return [self.paths[item][i] for i in range(3)]

    def __len__(self):
        return len(self.pairs)


def verification_label_loader():
    df = pd.read_csv(check_sys() + "/test_trials_verification_student.txt", header=None)
    df = df.rename(columns={0: "trial"})
    df["score"] = 0
    return df


def check_sys():
    cwd = os.getcwd()
    if "C:\\Users" in cwd:  # local env
        return "C:/Users/Jeffy/Downloads/Data/hw2/hw2p2_check"
    else:  # aws env
        return "/home/ubuntu/Data/hw2"


if __name__ == '__main__':
    self = VerificationTrain("train")
    self.__getitem__(100)
