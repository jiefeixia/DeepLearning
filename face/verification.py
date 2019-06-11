import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import sys
import time
import argparse

from model import *
from loader import *

"""###################################  init  ###################################"""
parser = argparse.ArgumentParser(description='11785 hw2p2')
parser.add_argument('--model', "-m", type=str, default="ResNet50", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--verification', action='store_true', help='further tuning verification model')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--pretrain', type=str, help='load pretrained model')
parser.add_argument('--predict', action='store_true', help='use existing model to preditc on test data')
parser.add_argument('--epoch',"-e", default=10, type=int, help='max epoch')
args = parser.parse_args()


def init():
    global net
    global model_stamp

    if args.model == "ResNet50":
        net = ResNet(Bottleneck, [3,4,6,3],  num_classes=num_classes)
    elif args.model == "ResNet18":
        net = ResNet(BasicBlock, [2, 2, 2, 2])
    else:
        net = None
        print("no specific model")
        sys.exit(0)

    if args.pretrain:
        print("loading pretrained model from", args.pretrain)
        pretrained_dict = torch.load(args.pretrain)["net"]
        net_dict = net.state_dict()
        net_dict.update({k: v for k, v in pretrained_dict.items() if k in net_dict})
        net.load_state_dict(net_dict)
        model_stamp = args.pretrain[:-6] + "_veri"

    elif args.resume or args.predict:
        print("loading exist model from %s" % args.resume)
        check_point = torch.load(args.resume)
        net.load_state_dict(check_point["net"])
        model_stamp = args.resume[:-4]

    net = net.cuda()


"""###################################  data loader  ###################################"""


def data_loader():
    global train_loader, val_loader, test_loader

    print("loading data...")
    val_dataset = VerificationTrain("validation")
    test_dataset = VerificationTest()

    if args.debug or args.predict:
        print("loading train dataset as the small validation dataset...")
        train_dataset = val_dataset
    else:
        train_dataset = VerificationTrain("train")

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             num_workers=8,
                             shuffle=False)


"""###################################  train  ###################################"""


def train(epoch, writer):
    global net, optimizer, criterion, train_loader
    net.train()

    running_loss = 0.0
    scores = torch.FloatTensor().cuda()
    labels = []
    auc = 0.0

    with tqdm(total=int(len(train_loader))) as pbar:
        for batch_idx, (x, x_pos, x_neg) in enumerate(train_loader):
            x, x_pos, x_neg = x.cuda(), x_pos.cuda(), x_neg.cuda()
            optimizer.zero_grad()

            outputs = net(x)
            outputs_pos = net(x_pos)
            outputs_neg = net(x_neg)

            loss = criterion(outputs, outputs_pos, outputs_neg)
            running_loss += loss.item()

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            scores = torch.cat((scores, cos(outputs, outputs_pos).detach()))
            labels += [1] * outputs_pos.shape[0]
            scores = torch.cat((scores, cos(outputs, outputs_neg).detach()))
            labels += [0] * outputs_neg.shape[0]

            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train Loss", loss.item(), niter)
                auc = roc_auc_score(labels, scores.cpu().numpy())
                pbar.set_postfix(T_current_loss=round(loss.item(), 4),
                                 T_avg_auc=round(auc, 4))

                pbar.update(10 if pbar.n + 10 <= pbar.total else pbar.total - pbar.n)

    running_loss /= len(train_loader)

    return running_loss, auc


def validate(epoch):
    global net, optimizer, criterion, val_loader
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        scores = torch.FloatTensor().cuda()
        labels = []

        for batch_idx, (x, x_pos, x_neg) in enumerate(val_loader):
            x, x_pos, x_neg = x.cuda(), x_pos.cuda(), x_neg.cuda()
            optimizer.zero_grad()

            outputs = net(x)
            outputs_pos = net(x_pos)
            outputs_neg = net(x_neg)

            loss = criterion(outputs, outputs_pos, outputs_neg)
            running_loss += loss.item()

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            scores = torch.cat((scores, cos(outputs, outputs_pos).detach()))
            labels += [1] * x.shape[0]
            scores = torch.cat((scores, cos(outputs, outputs_neg).detach()))
            labels += [0] * x.shape[0]

        running_loss /= (batch_idx + 1)
        return running_loss, roc_auc_score(labels, scores.cpu().numpy())


def run_epochs():
    epoch = 0
    n_epochs = args.epoch
    if args.resume:
        check_point = torch.load(args.resume)
        epoch = check_point["epoch"] + 1
    elif args.predict:
        epoch = 10
    elif args.debug:
        n_epochs = 1

    writer = SummaryWriter("log/%s" % model_stamp)

    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []

    print("start training from", epoch, "to", n_epochs)
    for e in range(epoch, n_epochs):
        if args.annealing:
            scheduler.step()

        train_loss, train_auc = train(epoch, writer)
        train_losses.append(train_loss)
        train_aucs.append(train_auc)

        val_loss, val_auc = validate(epoch)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        print("\re %3d: |Train avg loss: %.3f|Train avg auc: %.3f|Val avg loss: %.3f|Val avg auc: %.3f" %
              (e, train_loss, train_auc, val_loss, val_auc))

        # save check point
        if not args.debug:
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)

    writer.close()
    return train_losses, val_losses, train_aucs, val_aucs


"""###################################  predict  ###################################"""


def prediction():
    print("predicting with %s..." % model_stamp)
    with torch.no_grad():
        net.eval()
        scores = torch.FloatTensor().cuda()
        for batch_idx, (x1, x2) in enumerate(tqdm(test_loader)):
            x1, x2 = x1.cuda(), x2.cuda()
            y1 = net(x1)
            y2 = net(x2)

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            scores = torch.cat((scores, cos(y1, y2).detach()))

    return scores.cpu().numpy()


def save(pred, train_losses, val_losses, train_aucs, val_aucs):
    global model_stamp
    print("saving predicted result in ./result/%s.csv" % model_stamp)

    plt.plot(train_losses, val_losses)
    plt.savefig("result/" + model_stamp + ' loss.png')
    plt.plot(train_aucs, val_aucs)
    plt.savefig("result/" + model_stamp + " auc.png")

    df = verification_label_loader()
    df["score"] = pred

    df.to_csv("result/%s.csv" % model_stamp, index=False)




"""###################################  main  ###################################"""
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    init()

    data_loader()

    global criterion, optimizer, scheduler
    criterion = nn.TripletMarginLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    train_losses, val_losses, train_aucs, val_accs = run_epochs()

    pred = prediction()

    save(pred, train_losses, val_losses, train_aucs, val_accs)
