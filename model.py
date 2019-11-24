
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class baseline(nn.Module):
    def __init__(self, num_class,type):

        super(baseline, self).__init__()
        self.type = 'not_buttons'
        # self.fc1 = nn.Linear(100*100,40*40)
        self.fc2 = nn.Linear(100*100, num_class)


    def forward(self, x,loss_fnc = 'MSE'):
        x=x.mean(1)
        if self.type == 'buttons':
            x = x.reshape((x.size()[0],256*256))
            if loss_fnc == 'MSE':
                x = F.sigmoid(x)
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

    def __init__(self, num_class, batch_norm =False, dropout = 0.0, n1 = 5, n2 = 8, k1 = 3, k2 = 5, l1= 40, type = 'nobutton'):
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
                x = F.sigmoid(x)
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
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()

        return x

class cnn_buttons(nn.Module):

    def __init__(self, num_class=1, batch_norm =True, dropout = 0.2, n1 = 5, n2 = 8, k1 = 3, k2 = 5, l1= 40, type = 'buttons'):
        super(cnn_buttons,self).__init__()
        self.type = type
        if type == 'buttons':
            self.L = int((256 - (k1 - k1%2))/2)
            self.L = int((self.L - (k2 - k2%2))/2)
            self.L = self.L*self.L*n2
        else:
            self.L = int((100 - (k1 - k1 % 2)) / 2)
            self.L = int((self.L - (k2 - k2 % 2)) / 2)
            self.L = self.L * self.L * n2

        super(cnn,self).__init__()
        self.conv1 = nn.Conv2d(3,8,5)
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


    def forward(self, x, batch_norm=True, loss_fnc = 'CE'):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))

            x = x.view(-1, self.L)
            x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = F.sigmoid(x)
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
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()

        return x

class cnn3(nn.Module):
    def __init__(self, num_class, batch_norm=False, dropout=0.0, n1=5, n2=8, k1=3, k2=5,n3 = 5,k3=3, l1=40, type='nobutton'):
        super(cnn3, self).__init__()
        self.type = type
        if type == 'buttons':
            self.L = int((256 - (k1 - k1 % 2)) / 2)
        else:
            self.L = int((100 - (k1 - k1 % 2)) / 2)
        self.L = int((self.L - (k2 - k2 % 2)) / 2)
        self.L = int((self.L - (k3 - k3 % 2)) / 2)
        self.L = self.L * self.L * n3

        self.conv1 = nn.Conv2d(3, n1, k1)
        self.conv2 = nn.Conv2d(n1, n2, k2)
        self.conv3 = nn.Conv2d(n2, n3, k3)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.L, l1)
        self.fc2 = nn.Linear(l1, num_class)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.conv_bn1 = nn.BatchNorm2d(n1)
            self.conv_bn2 = nn.BatchNorm2d(n2)
            self.conv_bn3 = nn.BatchNorm2d(n3)
            self.bn1 = nn.BatchNorm1d(l1)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False, loss_fnc = 'MSE'):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))
            x = self.pool(F.relu(self.conv_bn3(self.dropout2(self.conv3(x)))))
            x = x.view(-1, self.L)
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.bn2(self.fc2(x))
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = self.pool(F.relu(self.dropout2(self.conv3(x))))
            x = x.view(-1, self.L)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)

            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        return x

class cnn4(nn.Module):
    def __init__(self, num_class, batch_norm=False, dropout=0.0, n1=5, n2=8, k1=3, k2=5, n3=5, k3=3,k4=3,n4=3, l1=40,type='nobutton'):
        super(cnn4, self).__init__()
        self.type = type
        if type == 'buttons':
            self.L = int((256 - (k1 - k1 % 2)) / 2)
        else:
            self.L = int((100 - (k1 - k1 % 2)) / 2)
        self.L = int((self.L - (k2 - k2 % 2)) / 2)
        self.L = int((self.L - (k3 - k3 % 2)) / 2)
        self.L = int((self.L - (k4 - k4 % 2)) / 2)
        self.L = self.L * self.L * n4

        self.conv1 = nn.Conv2d(3, n1, k1)
        self.conv2 = nn.Conv2d(n1, n2, k2)
        self.conv3 = nn.Conv2d(n2,n3,k3)
        self.conv4 = nn.Conv2d(n3,n4,k4)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.L, l1)
        self.fc2 = nn.Linear(l1, num_class)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout2d(p=dropout)
        if batch_norm:
            self.conv_bn1 = nn.BatchNorm2d(n1)
            self.conv_bn2 = nn.BatchNorm2d(n2)
            self.conv_bn3 = nn.BatchNorm2d(n3)
            self.conv_bn4 = nn.BatchNorm2d(n4)
            self.bn1 = nn.BatchNorm1d(l1)
            self.bn2 = nn.BatchNorm1d(num_class)


    def forward(self, x, batch_norm=False, loss_fnc = 'MSE'):
        if (batch_norm):
            x = self.pool(F.relu(self.conv_bn1(self.dropout2(self.conv1(x)))))
            x = self.pool(F.relu(self.conv_bn2(self.dropout2(self.conv2(x)))))
            x = self.pool(F.relu(self.conv_bn3(self.dropout2(self.conv3(x)))))
            x = self.pool(F.relu(self.conv_bn4(self.dropout2(self.conv4(x)))))
            x = x.view(-1, self.L)
            x = F.relu(self.bn1(self.dropout1(self.fc1(x))))
            x = self.bn2(self.fc2(x))
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        else:
            x = self.pool(F.relu(self.dropout2(self.conv1(x))))
            x = self.pool(F.relu(self.dropout2(self.conv2(x))))
            x = self.pool(F.relu(self.dropout2(self.conv3(x))))
            x = self.pool(F.relu(self.dropout2(self.conv4(x))))
            x = x.view(-1, self.L)
            x = F.relu(self.dropout1(self.fc1(x)))
            x = self.fc2(x)
            if loss_fnc == 'MSE' and self.type != 'buttons':
                x = F.softmax(x, dim=1)
            elif loss_fnc == 'MSE' and self.type == 'buttons':
                x = F.sigmoid(x)
            if self.type == 'buttons':
                x = x.squeeze()
        return x
