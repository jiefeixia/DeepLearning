import torch
import numpy as np
from dataloader import *
import torch.nn.functional as F


class Node(object):
    def __init__(self, parent, char, char_log_prob, attention_weight, cells):
        super(Node, self).__init__()
        self.parent = parent
        self.char = char
        self.char_log_prob = char_log_prob
        self.cells = cells
        if parent is None:
            self.attention_weights = [attention_weight]
            self.seq_log_prob = char_log_prob
            self.seq = idx2chr[char]
        else:
            self.attention_weights = parent.attention_weights
            self.attention_weights.append(attention_weight)
            self.seq_log_prob = parent.seq_log_prob + char_log_prob
            self.seq = parent.seq + idx2chr[char] if char < 34 else parent.seq


def beam_search(init_state, time_step, listener_hiddens, listener_h, listener_c, beam_width, max_len):
    """
    :param init_state:
    :param time_step:
    :param listener_hiddens: (1, listener L, listener H)
    :param listener_h: (1, layers, listener H)
    :param listener_c: (1, layers, listener H)
    :param beam_width:
    :param max_len:
    :return:
    """
    char, h1, c1, h2, c2 = init_state(listener_h, listener_c)
    listener_len = torch.tensor(listener_hiddens.shape[1]).cuda()

    nodes = [None] * beam_width  # NL(nodes length)=1
    listener_hiddens = listener_hiddens.repeat(beam_width, 1, 1)
    listener_len = listener_len.repeat(beam_width)
    h1 = h1.repeat(beam_width, 1)
    c1 = c1.repeat(beam_width, 1)
    h2 = h2.repeat(beam_width, 1)
    c2 = c2.repeat(beam_width, 1)
    char = char.repeat(beam_width)
    for t in range(max_len):
        char_log_prob, h1, c1, h2, c2,  attention_weight = time_step(listener_hiddens, listener_len, h1, c1, h2, c2, char)
        """
        listener_hiddens (B, listener padded L, listener H),
        listener_len(B,),
        h(B, speller H),
        c(B, speller H), char(last NL,)

        char_prob (B, C+1),
        h(B, speller H),
        c(B, speller H),
        attention_weight(B, listener padded L)
        """

        char_prob = F.gumbel_softmax(char_log_prob, tau=1)
        nodes = np.array([Node(parent=parent,
                               char=char,
                               char_log_prob=char_prob[i, char],
                               attention_weight=attention_weight[i],
                               cells=(h1[i], c1[i], h2[i], c2[i]))
                          for i, parent in enumerate(nodes) for char in range(34)])
        # NL =  B * C

        seq_log_prob = torch.tensor([node.seq_log_prob for node in nodes])
        seq_log_prob_topk, seq_topk = seq_log_prob.topk(k=beam_width, sorted=False)
        nodes = nodes[seq_topk]
        # if over half of beam predict <eos>, then break
        eos_count = torch.tensor([node.char == chr2idx[">"] for node in nodes]).sum()
        pad_count = torch.tensor([node.char == 35 for node in nodes]).sum()
        if eos_count > len(nodes) // 2 or pad_count > len(nodes) // 2:
            break

        char = torch.tensor([node.char for node in nodes]).cuda()
        h1 = torch.stack([node.cells[0] for node in nodes])
        c1 = torch.stack([node.cells[1] for node in nodes])
        h2 = torch.stack([node.cells[2] for node in nodes])
        c2 = torch.stack([node.cells[3] for node in nodes])

    max_seq_log_prob = - 1e5
    for node in nodes:
        if node.seq_log_prob > max_seq_log_prob:
            seq = node.seq.replace("<", "").replace(">", "")
            attention_weights = torch.stack(node.attention_weights, dim=0)

    return seq, attention_weights
