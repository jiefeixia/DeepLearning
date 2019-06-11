import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import sys
import time
import argparse

from model import *
from loader import *
from phoneme_list import PHONEME_MAP

import ctcdecode
import Levenshtein as L

"""###################################  init  ###################################"""
parser = argparse.ArgumentParser(description='11785 hw2p2')
parser.add_argument('--model', "-m", type=str, default="BiLSTM", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='use existing model to preditc on test data')
parser.add_argument('--large', action='store_true', help='use large dataset to tune model')
parser.add_argument('--init_xavier', '-i', action='store_true', help='init with xavier')
parser.add_argument('--pretrain', type=str, help='load pretrained model')
parser.add_argument('--epoch', "-e", default=10, type=int, help='max epoch')
parser.add_argument('--center_weight', "-c", default=0.1, type=float, help='weight for center loss')
args = parser.parse_args()


def init():
    global net, model_stamp

    if args.model == "BiLSTM":
        net = BiRNN(input_size=40, hidden_size=128, num_layers=2, num_classes=47)

    if args.resume or args.predict:
        print("loading exist model from %s" % args.resume)
        check_point = torch.load(args.resume)
        net.load_state_dict(check_point["net"])
        model_stamp = args.resume[:-4]
    else:
        t = time.localtime()
        model_stamp = "%s_%d_%d" % (args.model, t.tm_mday, t.tm_hour)

    if args.init_xavier:
        net.apply(xavier)

    print("initializing " + model_stamp)
    net = net.cuda()

    PHONEME_MAP.insert(0, ".")


"""###################################  data loader  ###################################"""


def padding_collate(batch):
    """
    :param batch: [(X(len, in_chan), y(len)) * batch_size]
    :return: [X(padded_len, batch_size, in chan), y(sum(len)]
    """
    input_list = [torch.from_numpy(x[0]) for x in batch]
    label_list = [x[1] for x in batch]
    zip_list = zip(input_list, label_list)

    sorted_list = sorted(zip_list, key=lambda x: len(x[0]), reverse=True)
    to_be_packed = [x[0] for x in sorted_list]

    input_len = torch.IntTensor([len(x[0]) for x in sorted_list])
    batch_input = rnn.pad_sequence(to_be_packed)
    lens = [len(x) for x in to_be_packed]
    target_len = torch.IntTensor([len(x[1]) for x in sorted_list])
    batch_input = rnn.pack_padded_sequence(batch_input, lens)

    y = torch.LongTensor([])
    for x in sorted_list:
        y = torch.cat((y, torch.from_numpy(x[1])))

    return batch_input, input_len, y+1, target_len


def padding_collate_test(batch):
    """
    :param batch: X(batch, len, in_chan)
    """
    input_len = torch.tensor([x.shape[0] for x in batch])
    sorted_input_len, sorted_idx = input_len.sort(descending=True)
    sorted_idx, seq_order = sorted_idx.sort()
    X = [batch[i] for i in sorted_idx]
    return X, sorted_input_len, seq_order


def data_loader():
    global train_loader, val_loader, test_loader

    print("loading data...")
    test_loader = DataLoader(TestData(),
                             batch_size=args.batch_size,
                             num_workers=2,
                             collate_fn=padding_collate_test,
                             shuffle=False)

    if not args.predict:
        val_dataset = TrainData("validation")
        if args.debug:
            print("loading train dataset as the small validation dataset...")
            train_dataset = val_dataset
            args.batch_size = 16
        else:
            train_dataset = TrainData("train")

        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=2,
                                shuffle=False,
                                collate_fn=padding_collate)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=2,
                                  shuffle=True,
                                  collate_fn=padding_collate)


"""###################################  train  ###################################"""


def decode(log_prob, input_len, catted_target=None, target_len=None):
    decoder = ctcdecode.CTCBeamDecoder(PHONEME_MAP,
                                       beam_width=100,
                                       blank_id=0,
                                       log_probs_input=True,
                                       num_processes=16)
    if catted_target is not None:  # calculate levenshtein distance
        output, scores, timesteps, out_seq_len = decoder.decode(log_prob, input_len)
        y_start = 0
        running_dist = []
        for i in range(output.size(0)):
            pred_str = "".join(PHONEME_MAP[f] for f in output[i, 0, :out_seq_len[i, 0]])
            label_str = "".join(PHONEME_MAP[f + 1] for f in catted_target[y_start:y_start + target_len[i]])
            running_dist.append(L.distance(pred_str, label_str))
            y_start += target_len[i]
            if i % 50 == 0:
                print("%s -> %s" % (label_str, pred_str))
            break
        return running_dist
    else:  # only calculate decoded result
        output, scores, timesteps, out_seq_len = decoder.decode(log_prob, input_len)
        pred_str = []
        for i in range(output.size(0)):
            pred_str.append("".join(PHONEME_MAP[f] for f in output[i, 0, :out_seq_len[i, 0]]))
        return pred_str


def train(epoch, writer):
    global net, optimizer, criterion, train_loader
    net.train()

    running_loss = 0.0

    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for batch_idx, (x, input_len, y, target_len) in enumerate(train_loader):
            optimizer.zero_grad()

            x, y = x.cuda(), y.cuda()
            outputs = net(x, input_len)
            log_prob = F.softmax(outputs, dim=2).log()
            loss = criterion(log_prob, y, input_len, target_len)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            torch.cuda.empty_cache()
            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train Loss", loss.item(), niter)
                pbar.set_postfix(T_current_loss=round(loss.item(), 4))

                pbar.update(10 if pbar.n + 10 <= pbar.total else pbar.total - pbar.n)

    running_loss /= len(train_loader)

    return running_loss


def validate(epoch):
    global net, optimizer, criterion, val_loader
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        running_dist = []

        for batch_idx, (x, input_len, y, target_len) in enumerate(val_loader):
            y, input_len, target_len = y.cuda(), input_len.cuda(), target_len.cuda()  # x(padded_len, batch, in_chan) y(sum(len))
            x = pad_sequence(x).cuda()
            outputs = net(x, input_len)
            log_prob = outputs.log_softmax(2)
            loss = criterion(log_prob, y, input_len, target_len)
            loss = torch.mean(loss)
            running_loss += loss.item()

            log_prob = log_prob.permute(1, 0, 2)
            dist = decode(log_prob, input_len, y, target_len)
            running_dist += dist

        running_loss /= len(val_loader)
        running_dist = np.average(running_dist)
        return running_loss, running_dist


def run_epochs():
    epoch = 0
    if args.resume:
        check_point = torch.load(args.resume)
        epoch = check_point["epoch"] + 1
    if args.predict:
        epoch = 10
    elif args.debug:
        args.epoch = 1

    writer = SummaryWriter("log/%s" % model_stamp)

    print("start training from epoch", epoch, "-", args.epoch)
    best_val_dist = 1000
    train_losses = []
    val_losses = []
    for e in range(epoch, args.epoch):
        if args.annealing:
            scheduler.step()

        train_loss = train(epoch, writer)
        val_loss, val_dist = validate(epoch)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("\re %3d: |Train avg loss: %.3f|Val avg loss: %.3f|Val dist: %.3f" %
              (e, train_loss, val_loss, val_dist))

        # save check point
        if not args.debug and val_dist < best_val_dist:
            best_val_dist = val_dist
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)

    writer.close()
    return train_losses, val_losses


"""###################################  predict  ###################################"""


def predict():
    print("predicting with %s..." % model_stamp)
    with torch.no_grad():
        global net, test_loader
        net.eval()
        prediction = []
        for batch_idx, (x, input_len, seq_order) in enumerate(tqdm(test_loader)):
            input_len, seq_order = input_len.cuda(), seq_order.cuda()
            x = x.cuda()
            outputs = net(x, input_len)
            log_prob = outputs.log_softmax(2)

            log_prob = log_prob.gather(1, seq_order)
            log_prob = log_prob.permute(1, 0, 2)
            input_len = input_len.gather(1, seq_order)
            string = decode(log_prob, input_len)
            prediction += string
    return prediction


def save(pred, train_losses, val_losses):
    global model_stamp
    print("saving predicted result in ./result/%s.csv" % model_stamp)

    plt.plot(train_losses, val_losses)
    plt.savefig("result/" + model_stamp + ' loss.png')

    df = pd.DataFrame({"id": [], "label": []})
    df["Predicted"] = pred
    df["Id"] = df.index

    df.to_csv("result/%s.csv" % model_stamp, index=False)


"""###################################  main  ###################################"""
if __name__ == '__main__':
    init()

    data_loader()  # return train and test dataset to produce prediction

    global criterion, optimizer, scheduler
    criterion = nn.CTCLoss(blank=0, reduction="none")
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = run_epochs()

    pred = predict()

    save(pred, train_losses, val_losses)
