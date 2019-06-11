import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from model import AngleLinear


class AngleLoss(nn.Module):
    def __init__(self):
        super(AngleLoss, self).__init__()
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        x_cos, x_phi = input
        target = target.view(-1, 1)  # size=(B,1)

        index = torch.zeros(x_cos.shape).cuda().scatter_(1, target, 1).byte().cuda()  # byte index for target
        self.lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.01 * self.it))
        # only calculate the balanced x.phi(theta) on the target index
        x_phi_balanced = x_cos[index] * self.lamb / (1 + self.lamb) \
                         + x_phi[index] / (1 + self.lamb)  # size=(B,class_num)
        x_cos[index] = x_phi_balanced
        loss = torch.logsumexp(x_cos, dim=1) - x_phi_balanced
        loss = loss.mean()

        # index = cos_theta.data * 0.0  # size=(B,class_num)
        # index.scatter_(1, target.data.view(-1, 1), 1)
        # index = index.byte()
        # index = Variable(index)
        # output = cos_theta * 1.0  # size=(B,class_num)
        # output[index] -= cos_theta[index] * (1.0 + 0) / (1 + self.lamb)
        # output[index] += phi_theta[index] * (1.0 + 0) / (1 + self.lamb)

        # logpt = F.log_softmax(output, dim=1)
        # logpt = logpt.gather(1, target)
        # logpt = logpt.view(-1)
        # pt = Variable(logpt.data.exp())
        #
        # loss = -1 * (1 - pt) ** self.gamma * logpt
        # loss = loss.mean()

        return loss


if __name__ == '__main__':
    layer = AngleLinear(512, 2300)
    criterion = AngleLoss()
    output = layer(torch.randn((128, 512)))
    target = torch.randint(0, 2300, (1, 128))
    loss = criterion(output, target)
