
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


    def forward(self, x, batch_norm = False):
        x=x.mean(1)
        x = x.reshape((x.size()[0],100*100))
        # x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
        return x


class cnn(nn.Module):

    def __init__(self, num_class, batch_norm =False, dropout = 0.0):
        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,8,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,8,5)
        self.fc1 = nn.Linear(8*22*22,40)
        self.fc2 = nn.Linear(40,num_class)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm == True:
            self.conv_bn1 = nn.BatchNorm2d(8)
            self.conv_bn2 = nn.BatchNorm2d(8)
            self.bn1 = nn.BatchNorm1d(40)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False, dropout = 0.0):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropoout2(self.conv2(x)))))

            x = x.view(-1, 8*22*22)
            x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            x = F.softmax(x, dim=1)
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = x.view(-1, 8 * 22 * 22)
            x = F.relu(self.dropout1(self.fc1(x)))
            x = self.fc2(x)
            x = F.softmax(x, dim=1)

        return x


class cnn3(nn.Module):
    def __init__(self, num_class):

        super(cnn3, self).__init__()
        self.conv1 = nn.Conv2d(3, 4, 3)
        self.conv_bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv_bn2 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(4, 8, 5)
        self.conv3 = nn.Conv2d(8,10,5)
        self.conv_bn3 = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(10 * 9 * 9, 40)
        self.bn1 = nn.BatchNorm1d(40)
        self.fc2 = nn.Linear(40, num_class)
        self.bn2 = nn.BatchNorm1d(num_class)

    def forward(self, x, batch_norm=False):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.conv1(x))))
            x = self.pool(F.relu(self.conv_bn2(self.conv2(x))))
            x = self.pool(F.relu(self.conv_bn3(self.conv3(x))))
            x = x.view(-1, 10 *9* 9)
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.bn2(self.fc2(x))
            x = F.softmax(x, dim=1)
        else:
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 10 *9 *9)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
        return x