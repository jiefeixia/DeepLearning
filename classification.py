import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.models as models

from tqdm import tqdm
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import sys
import time
import argparse

from model import *
from loader import *
from loss import AngleLoss, CenterLoss

"""###################################  init  ###################################"""
parser = argparse.ArgumentParser(description='11785 hw2p2')
parser.add_argument('--model', "-m", type=str, default="ResNet50", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='use existing model to preditc on test data')
parser.add_argument('--large', action='store_true', help='use large dataset to tune model')
parser.add_argument('--init_xavier', '-i', action='store_true', help='init with xavier')
parser.add_argument('--pretrain', type=str, help='load pretrained model')
parser.add_argument('--epoch',"-e", default=10, type=int, help='max epoch')
parser.add_argument('--center_weight',"-c", default=0.1, type=float, help='weight for center loss')
args = parser.parse_args()


def init():
    global net
    global model_stamp

    num_classes = 2000 if args.large else 2300
    if args.model == "CenterResNet":
        net = CenterResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    elif args.model == "ResNet34":
        net = ResNet50(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    elif args.model == "ResNet18":
        net = ResNet50(BasicBlock, [2, 2, 2, 2],  num_classes=num_classes)
    elif args.model == "ResNet50":
        net = ResNet(Bottleneck, [3,4,6,3],  num_classes=num_classes)
        # net = models.resnet50(num_classes=num_classes)
    elif args.model == "Sphere":
        net = Sphere20a(num_classes=num_classes)
    else:
        net = None
        print("no specific model")
        sys.exit(0)

    if args.resume or args.predict:
        print("loading exist model from %s" % args.resume)
        check_point = torch.load(args.resume)
        net.load_state_dict(check_point["net"])
        model_stamp = args.resume[:-4]
    else:
        t = time.localtime()
        model_stamp = "%s_lr%f_%d_%d_clas" % (args.model, args.lr, t.tm_mday, t.tm_hour)

    def xavier(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            torch.nn.init.xavier_normal_(m.weight.data)

    if args.init_xavier:
        net.apply(xavier)

    if args.pretrain:
        print("loading pretrained model from", args.pretrain)
        pretrained_dict = torch.load(args.pretrain)["net"]
        pretrained_dict.pop("fc.weight", None)
        pretrained_dict.pop("fc.bias", None)
        net_dict = net.state_dict()
        net_dict.update({k: v for k, v in pretrained_dict.items() if k in net_dict})
        net.load_state_dict(net_dict)

        for name, param in net.named_parameters():
            if "fc" not in name:
                param.requires_grad = False

        model_stamp = "_large" if args.large else "_medium"

    print("initializing " + model_stamp)
    net = net.cuda()


"""###################################  data loader  ###################################"""


def data_loader():
    global train_loader, val_loader, test_loader

    print("loading data...")
    if args.large:
        val_dataset = ClassificationTrain("validation", large=True)
    else:
        val_dataset = ClassificationTrain("validation")

    test_dataset = ClassificationTrain("test")

    test_loader = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         num_workers=8,
                         shuffle=False)

    if args.debug or args.predict:
        print("loading train dataset as the small validation dataset...")
        train_dataset = val_dataset
    else:
        if args.large:
            train_dataset = ClassificationTrain("train", large=True)
        else:
            train_dataset = ClassificationTrain("train")

    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            num_workers=8,
                            shuffle=True)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=8,
                              shuffle=True)

    return train_dataset.classes, test_dataset.samples
    # global model_stamp
    # with open(model_stamp + "_train_classes.txt", "w") as f:
    #     f.write("\n".join(train_dataset.classes))


"""###################################  train  ###################################"""


def train(epoch, writer):
    global net, optimizer, criterion, train_loader, optimizer_centerloss, center_loss
    net.train()

    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    acc = 0.0

    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            outputs = net(x)

            if len(outputs) == 2:  #tuple
                optimizer_centerloss.zero_grad()
                
                features, outputs = outputs
                loss = center_loss(features, y) * args.center_weight + criterion(outputs, y)
                loss.backward()
                
                # multiple (1./args.center_weight) in order to remove the effect of args.center_weight on updating centers
                for param in center_loss.parameters():
                    param.grad.data *= (1./args.center_weight)
                optimizer_centerloss.step()

            else:
                loss = criterion(outputs, y)
                loss.backward()

            _, predicted = torch.max(outputs, 1)
            total_predictions += y.size(0)
            correct_predictions += (predicted == y).sum().item()

            running_loss += loss.item()

            optimizer.step()
            torch.cuda.empty_cache()

            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer.add_scalar("Train Loss", loss.item(), niter)
                acc = (correct_predictions / total_predictions)
                pbar.set_postfix(T_current_loss=round(loss.item(), 4),
                                 T_avg_acc=round(correct_predictions / total_predictions, 4))

                pbar.update(10 if pbar.n + 50 <= pbar.total else pbar.total - pbar.n)

    running_loss /= len(train_loader)

    return running_loss, acc


def validate(epoch):
    global net, optimizer, criterion, train_loader, optimizer_centerloss, center_loss
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0

        for batch_idx, (x, y) in enumerate(val_loader):
            x, y = x.cuda(), y.cuda()
            outputs = net(x)
            
            if len(outputs) == 2:  #tuple
                features, outputs = outputs
                loss = center_loss(features, y) * args.center_weight + criterion(outputs, y)
                loss = loss.detach()
                
                
                # multiple (1./args.center_weight) in order to remove the effect of args.center_weight on updating centers
                for param in center_loss.parameters():
                    param.grad.data *= (1./args.center_weight)
                
            else:
                loss = criterion(outputs, y).detach()

            _, predicted = torch.max(outputs, 1)
            total_predictions += y.size(0)
            correct_predictions += (predicted == y).sum().item()

            running_loss += loss.item()

        running_loss /= (batch_idx + 1)
        acc = (correct_predictions / total_predictions)
        return running_loss, acc


def run_epochs():
    epoch = 0
    n_epochs = 10
    if args.resume:
        check_point = torch.load(args.resume)
        epoch = check_point["epoch"] + 1
    if args.predict:
        epoch = 10
    elif args.debug:
        n_epochs = 1

    writer = SummaryWriter("log/%s" % model_stamp)

    if args.resume:
        train_losses = check_point["train_losses"]
        train_accs = check_point["train_accs"]
        val_accs = check_point["val_accs"]
        val_losses = check_point["val_losses"]
    else:
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

    if args.pretrain:
        print("training last layer for pre trained model")
        _, _ = train(0, writer)
        
        for name, param in net.named_parameters():
            if "fc" not in name:
                param.requires_grad = True

    print("start training from epoch", epoch, "-", args.epoch)
    best_val_acc = 0.
    for e in range(epoch, args.epoch):
        if args.annealing:
            scheduler.step()

        train_loss, train_acc = train(epoch, writer)
        val_loss, val_acc = validate(epoch)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print("\re %3d: |Train avg loss: %.3f|Train avg acc: %.3f|Val avg loss: %.3f|Val avg acc: %.3f" %
              (e, train_loss, train_acc, val_loss, val_acc))

        # save check point
        if not args.debug and val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "train_accs": train_accs,
                     'val_accs': val_accs,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)

    writer.close()
    return train_losses, val_losses, train_accs, val_accs


"""###################################  predict  ###################################"""


def prediction():
    print("predicting with %s..." % model_stamp)
    with torch.no_grad():
        global net, test_loader
        net.eval()
        pred = torch.LongTensor().cuda()
        for batch_idx, (x_test, _) in enumerate(test_loader):
            x_test = x_test.cuda()
            outputs = net(x_test)
            if len(outputs) == 2:  #tuple
                _, predicted = torch.max(outputs[0], 1)
            else:
                _, predicted = torch.max(outputs, 1)

            pred = torch.cat((pred, predicted))

    return pred.cpu().numpy()


def save(pred, train_losses, val_losses, train_accs, val_accs, classes, test_samples):
    global model_stamp
    print("saving predicted result in ./result/%s.csv" % model_stamp)

    plt.plot(train_losses, val_losses)
    plt.savefig("result/" + model_stamp + ' loss.png')
    plt.plot(train_accs, val_accs)
    plt.savefig("result/" + model_stamp + " acc.png")

    df = pd.DataFrame({"id": [], "label": []})
    df["label"] = [classes[idx] for idx in pred]
    df["id"] = [os.path.splitext(os.path.basename(path))[0] for path, _ in test_samples]
    df = df.sort_values(["id"], ascending=True)

    df.to_csv("result/%s.csv" % model_stamp, index=False)


"""###################################  main  ###################################"""
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    init()

    classes, test_samples = data_loader()  # return train and test dataset to produce prediction


    # if isinstance(net, CenterResNet):
    #     global optimizer_centerloss, center_loss
    #     center_loss = CenterLoss(num_classes=2300, feat_dim=512) 
    #     optimizer_centerloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)

    global criterion, optimizer, scheduler
    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
   

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses, train_accs, val_accs = run_epochs()

    pred = prediction()

    save(pred, train_losses, val_losses, train_accs, val_accs, classes, test_samples)
