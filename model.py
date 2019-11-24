
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchsummary import summary
import torch


class baseline(nn.Module):
    def __init__(self, num_class):

        super(baseline, self).__init__()
        # self.fc1 = nn.Linear(100*100,40*40)
        self.fc2 = nn.Linear(100*100, num_class)


    def forward(self, x):
        x=x.mean(1)
        if self.type == 'buttons':
            x = x.reshape((x.size()[0],256*256))
            if loss_fnc == 'MSE':
                x = torch.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        else:
            x = x.reshape((x.size()[0], 100 * 100))
            if loss_fnc == 'MSE':
                x = F.softmax(x, dim=1)
            if self.type == 'buttons':
                x = x.squeeze()

        return x


class cnn2(nn.Module):

    def __init__(self, num_class, batch_norm =False, dropout = 0.0, n1 = 5, n2 = 8, k1 = 3, k2 = 5, l1= 40, type = 'colors'):
        super(cnn2,self).__init__()
        self.type = type
        if type == 'buttons':
            self.L = int((256 - (k1 - k1%2))/2)
            self.L = int((self.L - (k2 - k2%2))/2)
            # print(self.L)
            self.L = self.L*self.L*n2
        else:
            self.L = int((100 - (k1 - k1 % 2)) / 2)
            self.L = int((self.L - (k2 - k2 % 2)) / 2)
            # print(self.L)
            self.L = self.L * self.L * n2

        """annie"""
        self.conv1 = nn.Conv2d(3,n1,k1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(n1,n2,k2)
        self.fc1 = nn.Linear(self.L,l1)
        self.fc2 = nn.Linear(l1,num_class)
        """"""

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm == True:
            self.conv_bn1 = nn.BatchNorm2d(n1)
            self.conv_bn2 = nn.BatchNorm2d(n2)
            self.bn1 = nn.BatchNorm1d(l1)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False, loss_fnc = 'MSE'):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))

            x = x.view(-1, self.L)
            x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = torch.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = x.view(-1,self.L)
            x = F.relu(self.dropout1(self.fc1(x)))
            x = self.fc2(x)
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = torch.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()

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