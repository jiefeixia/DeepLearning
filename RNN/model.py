import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, p=0):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout=p)

        # self.fc = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size),
        #                         nn.ReLU(),
        #                         nn.Linear(hidden_size, 64),
        #                         nn.ReLU(),
        #                         nn.Linear(64, num_classes))
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                if 'weight' in p:
                    torch.nn.init.orthogonal_(self.lstm.__getattr__(p))

    def forward(self, x):
        """

        :param x: (padded len, batch size, in channel), input_len: (batch_size) in descending
        :return:
        """
        batch_size = 64
        # padded_len = x.shape[0]
        # Set initial states
        # h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda()  # 2 for bidirection
        # c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).cuda()

        # x = pack_padded_sequence(x, input_len)
        out, _ = self.lstm(x)  # out: (padded_len, batch_size, hidden_size*2)
        out, input_len = pad_packed_sequence(out)

        # Decode the hidden state of the last time step
        out = self.fc(out.view(-1, self.hidden_size * 2))
        return out.view(-1, batch_size, self.num_classes)


