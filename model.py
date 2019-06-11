import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes=2300
channel_base = 64


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, use_batchnorm = True):
        
        super(Bottleneck, self).__init__()

        # [1,3,1] stride structure
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1)
        if(use_batchnorm):
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        # cross layer connection here
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # add operation here
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feat_dim=0, num_classes=num_classes):

        super(ResNet, self).__init__()
        self.in_planes = channel_base

        self.conv1 = nn.Conv2d(3, channel_base, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channel_base)

        self.layer1 = self.blocks(block, channel_base, num_blocks[0], stride=1)
        self.layer2 = self.blocks(block, channel_base*2, num_blocks[1])
        self.layer3 = self.blocks(block, channel_base*4, num_blocks[2])
        self.layer4 = self.blocks(block, channel_base*8, num_blocks[3])
        # this is for cross entropy loss
        self.fc = nn.Linear(block.expansion*channel_base*8, num_classes)
        # this is for center loss
        # self.linear_closs = nn.Linear(block.expansion*channel_base*8, feat_dim)

    def blocks(self, block, planes, num_blocks, stride = 2):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, use_closs = False, verification = False):

        res1 = F.relu(self.bn1(self.conv1(x)))
        # print("relu", res1.shape)
        res2 = self.layer1(res1)
        # print("1", res2.shape)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        out = F.avg_pool2d(res5, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        # if(verification == False):
        #     label_out = self.linear(out)
        # else:
        #     return out
        # if(use_closs):
        #     closs_out = self.linear_closs(out)
        #     return closs_out, label_out
        return out



def ResNet50(feat_dim):
    return ResNet(Bottleneck, [3,4,6,3])



def test():
    net = ResNet50()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

#test()