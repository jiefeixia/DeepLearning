import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn
from tqdm import tqdm
# from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
import time
import argparse

from model import *
from beam_search import beam_search
import Levenshtein as L

"""###################################  init  ###################################"""
parser = argparse.ArgumentParser(description='11785 hw2p2')
parser.add_argument('--model', "-m", type=str, default="LAS", help='Choose model structure')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--annealing', action='store_true', help='annealing')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--debug', action='store_true', help='debug mode with small dataset')
parser.add_argument('--resume', '-r', type=str, help='resume from checkpoint')
parser.add_argument('--predict', action='store_true', help='use existing model to preditc on test data')
parser.add_argument('--init_xavier', '-i', action='store_true', help='init with xavier')
parser.add_argument('--pretrain', type=str, help='load pretrained model')
parser.add_argument('--epoch', "-e", default=10, type=int, help='max epoch')
parser.add_argument('--clip', type=float, default=0.1, help='gradient clipping')
parser.add_argument('--stamp', type=str, default=None, help='model stamp')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay L2')
args = parser.parse_args()


def init():
    global net, model_stamp
    if args.model == "LAS":
        net = LAS(in_channel=40,
                  listener_hidden_size=256,
                  speller_hidden_size=512,
                  attention_size=128,
                  embedding_size=512)

    if args.resume or args.predict:
        print("loading exist model from %s" % args.resume)
        check_point = torch.load(args.resume)["net"]
        # check_point.pop("listener.decoder.weight", None)
        # check_point.pop("listener.decoder.bias", None)
        # net_dict = net.state_dict()
        # net_dict.update({k: v for k, v in check_point.items() if k in net_dict})
        # net.load_state_dict(net_dict)
        net.load_state_dict(check_point)
        model_stamp = args.resume[:-4]
    else:
        t = time.localtime()
        model_stamp = "%s_%d_%d" % (args.model, t.tm_mday, t.tm_hour)

    if args.init_xavier:
        for name, param in net.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

    if args.stamp is not None:
        model_stamp = args.stamp

    print("initializing " + model_stamp)
    net = net.cuda()


"""###################################  data loader  ###################################"""


def collate(batch):
    """
    :param batch: [(utter(len), transcript(classes)) * batch_size]
    :return: utter(batch_size, padded utter L, in_channel), trans(batch_size, padded trans L)
    """
    utter, trans = zip(*batch)

    utter_len = torch.tensor([u.shape[0] for u in utter])
    trans_len = torch.tensor([t.shape[0] for t in trans])

    sorted_utter_len, sorted_idx = utter_len.sort(descending=True)

    utter = [torch.from_numpy(utter[i]) for i in sorted_idx]
    trans = [torch.from_numpy(trans[i]) for i in sorted_idx]
    trans_len = [trans_len[i] for i in sorted_idx]
    trans_len = torch.from_numpy(np.array(trans_len))

    utter = rnn.pad_sequence(utter, batch_first=True)
    trans = rnn.pad_sequence(trans, batch_first=True, padding_value=chr2idx[">"])

    return utter, sorted_utter_len, trans.long(), trans_len


def collate_test(utter):
    """
    :param utter:(batch, len, in_chan)
    """
    utter_len = torch.tensor([u.shape[0] for u in utter])
    sorted_utter_len, sorted_idx = utter_len.sort(descending=True)
    sorted_idx, seq_order = sorted_idx.sort()
    utter = [torch.from_numpy(utter[i]) for i in sorted_idx]
    utter = rnn.pad_sequence(utter, batch_first=True)
    return utter, sorted_utter_len, seq_order.numpy()


def data_loader():
    global train_loader, val_loader, test_loader

    print("loading data...")
    test_loader = DataLoader(TestData(),
                             batch_size=args.batch_size,
                             num_workers=1 if args.debug else 6,
                             collate_fn=collate_test,
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
                                num_workers=1 if args.debug else 6,
                                shuffle=False,
                                collate_fn=collate)

        train_loader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=1 if args.debug else 6,
                                  shuffle=True,
                                  collate_fn=collate)


"""###################################  train  ###################################"""


def decode(out, target=None, target_len=None):
    _, pred = torch.max(out, dim=2)
    if target is not None:  # calculate levenshtein distance
        running_dist = []
        max_dist = 0
        for i in range(pred.size(0)):
            pred_str = "".join(idx2chr[idx] for idx in pred[i])
            if ">" in pred_str:
                pred_str = pred_str[0:pred_str.index(">") + 1]
            label_str = "".join(idx2chr[idx] for idx in target[i, 0:target_len[i]])

            dist = L.distance(pred_str, label_str)
            running_dist.append(dist)
            if dist > max_dist:
                print("%s\n%s" % (label_str, pred_str))
                print("-----------------------------------------")
                max_dist = dist
        return running_dist

    else:  # only return decoded result
        pred_str = []
        for b in range(pred.size(0)):
            start = (pred[b, :] == chr2idx[">"]).nonzero()
            end = (pred[b, :] == chr2idx[">"]).nonzero()
            if start.shape[0] == 0:
                start = -1
            else:
                start = start[0][0]
            if end.shape[0] == 0:
                end = pred.shape[1]
            else:
                end = end[0][0]
            pred_str.append("".join(idx2chr[idx] for idx in pred[b, (start + 1):end]))
        return pred_str


def plot_grad_flow(named_parameters, figname):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(figname, bbox_inches='tight')
    plt.close()


def train(epoch, writer):
    global net, optimizer, criterion, train_loader
    net.train()

    running_loss = 0.0
    losses = []

    with tqdm(total=int(len(train_loader)), ascii=True) as pbar:
        for batch_idx, (utter, utter_len, trans, trans_len) in enumerate(train_loader):
            optimizer.zero_grad()

            utter, utter_len, trans, trans_len = utter.cuda(), utter_len.cuda(), trans.cuda(), trans_len.cuda()

            out, attention_weights, listener_len = net(utter, utter_len, trans, epoch)  # (N, T, C+1)

            # out = out.permute(1, 0, 2)  # (T, N, C+1)
            # loss = criterion(out, trans, trans_len, trans_len)

            # out = out.permute(0, 2, 1)  # (N, C+1, T)
            sorted_trans_len, sorted_idx = trans_len.sort(descending=True)
            out = out[sorted_idx, :, :]
            trans = trans[sorted_idx, :]
            out = pack_padded_sequence(out, sorted_trans_len, batch_first=True).data
            trans = pack_padded_sequence(trans, sorted_trans_len, batch_first=True).data
            loss = criterion(out, trans)
            loss = torch.mean(loss)
            # mask = torch.zeros(trans.shape, requires_grad=False).cuda()
            # for i in range(trans.shape[0]):
            #     mask[i, 0:trans_len[i]] = 1
            # loss = torch.sum(loss * mask) / torch.sum(trans_len)

            loss.backward()
            # gradient clipping
            # torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            # for p in net.parameters():
            #     p.data.add_(-args.lr, p.grad.data)
            optimizer.step()
            running_loss += loss.item()
            losses.append(loss.item())

            torch.cuda.empty_cache()
            if batch_idx % 10 == 0:
                niter = epoch * len(train_loader) + batch_idx
                # writer.add_scalar("Train Loss", loss.item(), niter)
                pbar.set_postfix(P=round(np.exp(loss.item()), 4))
                pbar.update(10 if pbar.n + 10 <= pbar.total else pbar.total - pbar.n)
                fig_freq = 10 if args.debug else 100
                if batch_idx % fig_freq == 0:
                    plot_grad_flow(net.named_parameters(),
                                   "result/%s_gf_train_e%d_b%d.png" % (model_stamp, epoch, batch_idx))
                    # print(utter_len[-1], listener_len[-1], trans_len[-1])
                    plt.imshow(attention_weights[-1].detach().cpu().numpy(),
                               interpolation='nearest',
                               cmap='hot')
                    plt.xlabel("listener L%d" % (listener_len[-1]))
                    plt.ylabel("speller L%d" % (trans_len[-1]))
                    plt.savefig("result/%s_aw_train_e%d_b%d.png" % (model_stamp, epoch, batch_idx))
                    plt.close()
    running_loss /= len(train_loader)

    return running_loss, losses


def validate(epoch):
    global net, optimizer, criterion, val_loader
    with torch.no_grad():
        net.eval()

        running_loss = 0.0
        running_dist = []

        for batch_idx, (utter, utter_len, trans, trans_len) in enumerate(val_loader):
            utter, utter_len, trans, trans_len = utter.cuda(), utter_len.cuda(), trans.cuda(), trans_len.cuda()

            out, attention_weights, listener_len = net(utter, utter_len, trans)

            dist = decode(out, trans, trans_len)

            sorted_trans_len, sorted_idx = trans_len.sort(descending=True)
            out = out[sorted_idx, :, :]
            trans = trans[sorted_idx, :]
            out = pack_padded_sequence(out, sorted_trans_len, batch_first=True).data
            trans = pack_padded_sequence(trans, sorted_trans_len, batch_first=True).data
            loss = criterion(out, trans)
            loss = torch.mean(loss)
            running_loss += loss.item()

            running_dist += dist
        plt.imshow(attention_weights[-1].detach().cpu().numpy(), interpolation='nearest', cmap='hot')
        plt.savefig("result/%s_aw_val_e%d.png" % (model_stamp, epoch))
        plt.close()
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

    # writer = SummaryWriter("log/%s" % model_stamp)
    writer = None

    print("start training from epoch", epoch, "-", args.epoch)
    best_val_dist = 1000
    train_losses = []
    batch_losses = []
    val_losses = []
    for e in range(epoch, args.epoch):
        if args.annealing:
            scheduler.step()

        train_loss, losses = train(e, writer)
        val_loss, val_dist = validate(e)

        train_losses.append(train_loss)
        batch_losses += losses
        val_losses.append(val_loss)

        print("\re %3d: |Train avg loss: %.3f|Val avg loss: %.3f|Val dist: %.3f" %
              (e, np.exp(train_loss), np.exp(val_loss), val_dist))
        plt.plot(batch_losses)
        plt.savefig("result/" + model_stamp + '_train_loss.png')
        plt.close()
        # save check point
        if val_dist < best_val_dist:
            print("saving best model...")
            best_val_dist = val_dist
            state = {'net': net.state_dict(),
                     "train_losses": train_losses,
                     "val_losses": val_losses,
                     'epoch': e,
                     }
            torch.save(state, '%s.pth' % model_stamp)

    # writer.close()
    return train_losses, val_losses


"""###################################  predict  ###################################"""


def predict_greedy():
    print("greedily predicting with %s..." % model_stamp)
    with torch.no_grad():
        global net, test_loader
        net.eval()
        prediction = []
        for batch_idx, (utter, utter_len, seq_order) in enumerate(test_loader):
            utter, utter_len, seq_order = utter.cuda(), utter_len.cuda(), seq_order.cuda()
            out = net(utter, utter_len)

            out = out[seq_order]
            string = decode(out)
            prediction += string
    return prediction


def predict_beam_search(beam_width):
    print("beam-search predicting with %s..." % model_stamp)
    with torch.no_grad():
        global net, test_loader
        net.eval()
        prediction = []
        for batch_idx, (utter, utter_len, seq_order) in enumerate(tqdm(test_loader)):
            utter, utter_len = utter.cuda(), utter_len.cuda()
            batch_size = utter.shape[0]
            (listener_hiddens, (listener_h, listener_c)), listener_len = net.listener(utter=utter, utter_len=utter_len)

            seqs = []
            for b in range(batch_size):
                seq, attention_weights = beam_search(init_state=net.speller.init_state,
                                                     time_step=net.speller.time_step,
                                                     listener_hiddens=listener_hiddens[b, :listener_len[b]].unsqueeze(
                                                         0),
                                                     listener_h=listener_h[b].unsqueeze(0),
                                                     listener_c=listener_c[b].unsqueeze(0),
                                                     beam_width=beam_width,
                                                     max_len=int(utter_len[b] / 6))
                seqs.append(seq)
            plt.imshow(attention_weights.cpu().numpy(), interpolation='nearest', cmap='hot')
            plt.savefig("result/%s_aw_test_b%d.png" % (model_stamp, batch_idx))
            seqs = np.array(seqs)
            seqs = seqs[seq_order]
            prediction += seqs.tolist()
    return prediction


def save(pred, train_losses, val_losses):
    global model_stamp
    print("saving predicted result in ./result/%s.csv" % model_stamp)

    df = pd.DataFrame({"Id": [], "Predicted": []})
    df["Predicted"] = pred
    df["Id"] = df.index

    df.to_csv("result/%s.csv" % model_stamp, index=False)


"""###################################  main  ###################################"""
if __name__ == '__main__':
    init()

    data_loader()  # return train and test dataset to produce prediction

    global criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_losses, val_losses = run_epochs()

    pred = predict_beam_search(beam_width=3)

    save(pred, train_losses, val_losses)
