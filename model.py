
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import torch

# class cnn(nn.Module):
#
#     def __init__(self,num_kern,NF):
#         super(cnn,self).__init__()
#         self.num_kern = num_kern
#         self.conv1 = nn.Conv2d(3,num_kern,3)
#         #self.conv_bn = nn.BatchNorm2d(num_kern)
#         self.conv2 = nn.Conv2d(num_kern, num_kern, 3)
#         self.conv3 = nn.Conv2d(num_kern,num_kern,3)
#         self.pool = nn.MaxPool2d(2,2)
#         self.fc1 = nn.Linear(num_kern*5*5, NF)
#         #self.bn1 = nn.BatchNorm1d(NF)
#         self.fc2 = nn.Linear(NF, 10)
#         #self.bn2 = nn.BatchNorm1d(10)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         #print(x.shape)
#         x = x.view(-1,self.num_kern*5*5)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         x = F.softmax(x, dim=1)
#         return x

class baseline(nn.Module):
    def __init__(self, dimensions):

        super(baseline, self).__init__()
        self.fc1 = nn.Linear(300,dimensions)

    def forward(self, x):
        x = F.softmax(F.relu(self.fc1(x)))
        return x


class cnn(nn.Module):

    def __init__(self, num_class):

        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,4,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(4,8,5)

        self.fc1 = nn.Linear(8*22*22,40)
        self.fc2 = nn.Linear(40,num_class)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,8*22*22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x