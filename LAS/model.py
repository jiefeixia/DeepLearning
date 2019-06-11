import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from dataloader import *
import matplotlib.pyplot as plt
from torch.autograd import Variable


class LAS(nn.Module):
    def __init__(self, in_channel, listener_hidden_size, speller_hidden_size, attention_size, embedding_size):
        super(LAS, self).__init__()

        self.listener = Listener(in_channel=in_channel,
                                 hidden_size=listener_hidden_size)

        self.speller = Speller(listener_hidden_size=listener_hidden_size,
                               hidden_size=speller_hidden_size,
                               tf_rate=0.9,
                               attention_size=attention_size,
                               embedding_size=embedding_size)

    def forward(self, utter, utter_len, ground_truth=None, epoch=None):
        """
        :param utter: (N, padded utter L, in_C)
        :param utter_len:
        :param ground_truth: (N, padded seq L, in_C)
        :param epoch used for changing teaching force rate
        :return:
        """

        (listener_hiddens, (listener_h, listener_c)), listener_len = self.listener(utter=utter,
                                                                                   utter_len=utter_len)

        char_outs, attention_weights, listener_len = self.speller(
            listener_out=(listener_hiddens, (listener_h, listener_c)),
            listener_len=listener_len,
            ground_truth=ground_truth,
            epoch=epoch)

        return char_outs, attention_weights, listener_len


class Listener(nn.Module):
    def __init__(self, in_channel, hidden_size):
        super(Listener, self).__init__()
        self.rnns = nn.ModuleList([nn.LSTM(in_channel, hidden_size, bidirectional=True, batch_first=True),
                                   nn.LSTM(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True),
                                   nn.LSTM(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True),
                                   nn.LSTM(hidden_size * 4, hidden_size, bidirectional=True, batch_first=True)])

    def forward(self, utter, utter_len):
        """
        :param utter: (N, padded L, in_C)
        :param utter_len:
        :return: x: (N, listener padded L, listener H)
        """
        batch_size = utter.shape[0]
        for l, rnn in enumerate(self.rnns):
            utter = pack_padded_sequence(utter, utter_len, batch_first=True)
            utter, (h_n, c_n) = rnn(utter)
            utter, utter_len = pad_packed_sequence(utter, batch_first=True)
            if l <= 2:
                padded_len = utter.size(1)
                # make input len even number
                if padded_len % 2 != 0:
                    utter = utter[:, 0:-1, :]
                    padded_len -= 1
                utter_len = utter_len // 2
                utter = utter.contiguous().view(batch_size, padded_len // 2, utter.size(2) * 2)

        h_n = h_n.permute(1, 0, 2).squeeze(0)
        c_n = c_n.permute(1, 0, 2).squeeze(0)
        return (utter, (h_n, c_n)), utter_len


class Speller(nn.Module):
    def __init__(self, listener_hidden_size, hidden_size, tf_rate, attention_size, embedding_size):
        super(Speller, self).__init__()
        self.tf_rate = tf_rate
        self.hidden_size = hidden_size
        chr_size = len(chr2idx)  # from dataloader.py

        # self.listener_speller1_state_link = nn.Linear(listener_hidden_size, hidden_size)
        # self.listener_speller2_state_link = nn.Linear(listener_hidden_size, hidden_size)

        self.attention = Attention(listener_hidden_size=listener_hidden_size,
                                   speller_hidden_size=hidden_size,
                                   attention_size=attention_size)

        self.cell1 = nn.LSTMCell(input_size=attention_size + embedding_size,
                                 hidden_size=hidden_size)
        self.cell2 = nn.LSTMCell(input_size=hidden_size,
                                 hidden_size=hidden_size)

        # Num classes = C + 1, in order to generate the class for padded value,
        # but this label index(34) is disabled in CrossEntropy Loss
        self.encoder = nn.Embedding(chr_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, chr_size)
        self.context_fc = nn.Linear(attention_size, chr_size)

        # init
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.weight = self.encoder.weight
        self.decoder.bias.data.fill_(0)
        self.sos = torch.tensor(chr2idx["<"])

    def forward(self, listener_out, listener_len, ground_truth=None, epoch=None):
        listener_hiddens, (listener_h, listener_c) = listener_out

        char, h1, c1, h2, c2 = self.init_state(listener_h, listener_c)
        char_outs = []
        attention_weights = []
        for t in range(ground_truth.shape[1]):
            char_prob, h1, c1, h2, c2, attention_weight = self.time_step(listener_hiddens,
                                                                         listener_len,
                                                                         h1,
                                                                         c1,
                                                                         h2,
                                                                         c2,
                                                                         char)
            # char_log_prob = F.gumbel_softmax(char_log_prob, tau=1)
            char = self.teaching_force(char_prob, ground_truth[:, t], epoch)

            char_outs.append(char_prob)
            attention_weights.append(attention_weight)

        char_outs = torch.stack(char_outs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)  # (N, speller padded L, listener padded L)
        return char_outs, attention_weights, listener_len

    def time_step(self, listener_hiddens, listener_len, h1, c1, h2, c2, char):
        """
        :param listener_hiddens: (N, listener padded L, listener H)
        :param listener_len:(N,)
        :param h1: (N, speller H)
        :param c1: (N, speller H)
        :param h2: (N, speller H)
        :param c2: (N, speller H)
        :param char: (N,)
        :return: char_log_prob (N, C+1)
        :return: h1(N, speller H)
        :return: c1(N, speller H)
        :return: h2(N, speller H)
        :return: c2(N, speller H)
        :return: attention weight (N, listener padded L)
        """
        char_embed = self.encoder(char)
        context, attention_weight = self.attention(listener_hiddens, h1, listener_len)
        cell_input = torch.cat([char_embed, context], dim=1)
        h1, c1 = self.cell1(cell_input, (h1, c1))
        h2, c2 = self.cell2(h1, (h2, c2))

        chars_h1 = self.decoder(h2)  # (N, C)
        chars_context = self.context_fc(context)
        chars_context = self.sampler(chars_context)
        # chars_h1 = F.gumbel_softmax(F.log_softmax(chars_h1, dim=1), tau=1)
        # chars_context = F.softmax(chars_context, dim=1)
        char_prob = chars_h1 + chars_context
        # char_log_prob = F.log_softmax(chars_prob_h1 + chars_prob_context, dim=1)
        return char_prob, h1, c1, h2, c2, attention_weight

    def teaching_force(self, char_prob, char_true, epoch):
        char = torch.max(char_prob, dim=1)[1]
        if self.training:
            tf_rate = self.tf_rate * 0.8 ** (epoch // 5)
            teacher_force = True if np.random.uniform() < tf_rate else False
            if teacher_force:
                char = char_true
        return char

    def init_state(self, listener_h, listener_c):
        # speller_h1 = self.listener_speller1_state_link(listener_h)
        # speller_c1 = self.listener_speller1_state_link(listener_c)
        # speller_h2 = self.listener_speller2_state_link(listener_h)
        # speller_c2 = self.listener_speller2_state_link(listener_c)
        batch_size = listener_h.shape[0]
        speller_hidden_size = self.hidden_size
        speller_h1 = torch.zeros((batch_size, speller_hidden_size)).cuda()
        speller_c1 = torch.zeros((batch_size, speller_hidden_size)).cuda()
        speller_h2 = torch.zeros((batch_size, speller_hidden_size)).cuda()
        speller_c2 = torch.zeros((batch_size, speller_hidden_size)).cuda()

        batch_size = listener_h.shape[0]
        char = self.sos.repeat(batch_size).cuda()
        return char, speller_h1, speller_c1, speller_h2, speller_c2

    def sampler(self, input):
        """
        Apply gumbel noise to the input to last MLP from h2
        :return:
        """
        noise = torch.randn(input.size()).cuda()
        # noise.add_(1e-9).log_().neg_()
        # noise.add_(1e-9).log_().neg_()
        # noise = Variable(noise)
        # x = (input + noise) / tau + temperature
        # x = F.softmax(x.view(input.size(0), -1))
        return input * (1 + noise)


class Attention(nn.Module):
    def __init__(self, listener_hidden_size, attention_size, speller_hidden_size):
        super(Attention, self).__init__()
        self.key_fc = nn.Linear(listener_hidden_size * 2, attention_size)
        self.query_fc = nn.Linear(speller_hidden_size, attention_size)
        self.value_fc = nn.Linear(listener_hidden_size * 2, attention_size)

    def forward(self, listener_hiddens, speller_state, listener_len):
        """
        :param listener_hiddens: (N, listener padded L, listener H)
        :param speller_state: (N, speller H)
        :param listener_len:
        :return: context(N, listener H), attention weight (N, listener padded L)
        """
        batch_size = listener_hiddens.shape[0]
        listener_padded_len = listener_hiddens.shape[1]
        listener_hidden_size = listener_hiddens.shape[2]

        listener_hiddens_2d = listener_hiddens.contiguous().view(batch_size * listener_padded_len, listener_hidden_size)
        # (N * listener padded L, listener H)

        key = self.key_fc(listener_hiddens_2d)  # (B * listener padded L, A)
        key = key.view(batch_size, listener_padded_len, -1)
        key = key.permute(0, 2, 1)  # (N, A, listener padded L)

        query = self.query_fc(speller_state)  # (N, A)
        query = query.unsqueeze(1)  # (N, 1, A)

        energy = torch.bmm(query, key)  # (N, 1, listener padded L)

        energy = energy.squeeze(1)  # (N, listener padded L)

        attention_weight = F.softmax(energy, dim=1)
        mask = torch.arange(attention_weight.shape[1]).cuda() < listener_len.long().cuda().view(-1, 1)
        # mask = torch.zeros(attention_weight.shape, requires_grad=False).cuda()
        # for i in range(listener_len.shape[0]):
        #     mask[i, 0:listener_len[i]] = 1
        attention_weight = attention_weight * mask.float()
        attention_weight = F.normalize(attention_weight, dim=1, p=1)
        attention_weight = attention_weight.unsqueeze(1)  # (N, 1, listener padded L)

        values = self.value_fc(listener_hiddens)
        context = torch.bmm(attention_weight, values)  # (N, 1, listener H)
        context = context.squeeze(1)
        attention_weight = attention_weight.squeeze(1)
        return context, attention_weight


if __name__ == '__main__':
    idx2chr = []
    chr2idx = dict()
    with open(os.path.join(check_sys_path() + "chr2idx.txt")) as f:
        for l in f:
            l = l.strip().split(":")
            chr2idx[l[0]] = int(l[1])
    with open(os.path.join(check_sys_path() + "idx2chr.txt")) as f:
        for l in f:
            idx2chr.append(l.strip())

    net = LAS(in_channel=40,
              listener_hidden_size=256,
              speller_hidden_size=512,
              attention_size=128,
              embedding_size=512)

    net = net.cuda()
    utter = torch.randn(128, 200, 40).cuda()
    utter_len = torch.from_numpy(np.random.randint(150, 200, 128)).cuda()
    trans_len = torch.from_numpy(np.random.randint(20, 100, 128)).cuda()
    trans = torch.randint(0, 34, (128, 100)).cuda()

    net.forward(utter, utter_len, ground_truth=trans, epoch=5)

    net.eval()
    net.forward(utter, utter_len, ground_truth=trans)

    a = torch.randn(2, 5)
    ax = plt.subplot(111)
    ax.bar(range(5), F.softmax(a, dim=1)[0], width=0.2)
    ax.bar(np.fromiter(range(5), dtype="float") + 0.2, F.gumbel_softmax(F.log_softmax(a, dim=1))[0], width=0.2)
    plt.show()
