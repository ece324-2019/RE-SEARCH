
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import torch


class baseline(nn.Module):
    def __init__(self, num_class, batch_norm =False, dropout = 0.0):

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
        self.hidden_dim = 8*18*18
        self.conv1 = nn.Conv2d(3,8,10)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,8,10)
        self.fc1 = nn.Linear(self.hidden_dim,40)
        self.fc2 = nn.Linear(40,num_class)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm == True:
            self.conv_bn1 = nn.BatchNorm2d(8)
            self.conv_bn2 = nn.BatchNorm2d(8)
            self.bn1 = nn.BatchNorm1d(40)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))

            x = x.view(-1, self.hidden_dim)
            x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            x = F.softmax(x, dim=1)
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = x.view(-1, self.hidden_dim)
            x = F.relu(self.dropout1(self.fc1(x)))
            x = self.fc2(x)
            x = F.softmax(x, dim=1)

        return x


class cnn3(nn.Module):
    def __init__(self, num_class, batch_norm =False,dropout = 0.0):

        super(cnn3, self).__init__()
        self.hidden_dim = 5*7*7
        self.conv1 = nn.Conv2d(3, 5, 7)
        self.conv2 = nn.Conv2d(5, 5, 7)
        self.conv3 = nn.Conv2d(5, 5, 7)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.hidden_dim, 40)
        self.fc2 = nn.Linear(40, num_class)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.conv_bn1 = nn.BatchNorm2d(4)
            self.conv_bn2 = nn.BatchNorm2d(8)
            self.conv_bn3 = nn.BatchNorm2d(10)
            self.bn1 = nn.BatchNorm1d(40)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))
            x = self.pool(F.relu(self.conv_bn3(self.dropout2(self.conv3(x)))))
            x = x.view(-1, self.hidden_dim)
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.bn2(self.fc2(x))
            x = F.softmax(x, dim=1)
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = self.pool(F.relu(self.dropout2(self.conv3(x))))
            x = x.view(-1, self.hidden_dim)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            x = F.softmax(x, dim=1)
        return x