
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import torch


class baseline(nn.Module):
    def __init__(self, num_class):

        super(baseline, self).__init__()
        # self.fc1 = nn.Linear(100*100,40*40)
        self.fc2 = nn.Linear(100*100, num_class)


    def forward(self, x):
        x=x.mean(1)
        x = x.reshape((x.size()[0],100*100))
        # x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class cnn(nn.Module):

    def __init__(self, num_class, dropout = 0.0):

        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,8,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,8,5)
        self.fc1 = nn.Linear(8*22*22,40)
        self.fc2 = nn.Linear(40,num_class)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.dropout2(self.conv1(x))))
        x = self.pool(F.relu(self.dropout2(self.conv2(x))))
        x = x.view(-1,8*22*22)
        x = F.relu(self.dropout1(self.fc1(x)))
        x = (self.fc2(x))
        x = F.softmax(x, dim=1)
        return x