import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import cnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter as sf
import time
import torch.utils.data as data
from sklearn.metrics import confusion_matrix

from PIL import Image
import os, sys
# torch.manual_seed(2)
""" Type of classifier """
""" 1) args.type = color """
""" 2) args.type = buttons """
""" 3) args.type = neck """
""" 4) args.type = sleeve """

""" Testing case: black, white, orange """
""" import photos with correct labels """
def get_mean_std(dataloader):
    mean = []
    std = []
    for i, data in enumerate(dataloader):
        inputs,labels = data
        for j in range(labels.shape[0]):
            mean += [torch.mean(inputs[j])]
            std += [torch.std(inputs[j])]
    return np.mean(mean),np.mean(std)

def fetch():
    dataset = torchvision.datasets.ImageFolder(root='./test_data_2', transform=transforms.Compose(
        [transforms.ToTensor()]))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=257)
    mean, std = get_mean_std(dataloader)
    print("Original mean:", mean)
    print("Original std:", std)

    dataset = torchvision.datasets.ImageFolder(root='./test_data_2', transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std))]))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    mean, std = get_mean_std(dataloader)
    print("Calibrated mean:", mean)
    print("Calibrated std:", std)

    return dataloader

def get_num(n):
    num_train = int(n * 0.75)
    num_valid = int(n * 0.15)
    num_test =n - num_train - num_valid
    return num_train, num_valid, num_test

class ImageDataset(data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        return self.X[index],self.y[index]

def split(dataloader,labels):
    # testing case:
    # labels should be type [['white',121],['black',109],['orange',27]]

    num = []
    stats = []
    for i in range(0,len(labels)):
        num += [[0,0,0]]
        stats += [[0,0,0]]

    for i in range(0,len(labels)):
        num[i][0],num[i][1],num[i][2] = get_num(labels[i][1])

    num_train = 0
    num_valid = 0
    num_test = 0
    for i in range(0,len(num)):
        num_train += num[i][0]
        num_valid += num[i][1]
        num_test += num[i][2]

    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []
    test_data = []
    test_labels = []

    for i, data in enumerate(dataloader):
        inputs, label = data
        j = int(label)

        if stats[j][0] < num[j][0]:
            stats[j][0] += 1
            train_data += [inputs.data[0]]
            train_labels += [label.data[0]]
        elif stats[j][1] < num[j][1]:
            stats[j][1] += 1
            valid_data += [inputs.data[0]]
            valid_labels += [label.data[0]]
        elif stats[j][2] < num[j][2]:
            stats[j][2] += 1
            test_data += [inputs.data[0]]
            test_labels += [label.data[0]]
        else:
            print(j)
            print(stats[j])
            print(num[j])


    print("\nInput Labels:")
    print(labels)

    print('\n# Train Data: ', num_train, len(train_data))
    print("# Train Labels: ",num_train, len(train_labels))
    print("# Validation Data:",num_valid, len(valid_data))
    print('# Validation Labels: ', num_valid,len(valid_labels))
    print("# Test Data: ", num_test,len(test_data))
    print("# Test Labels:", num_test,len(test_labels),'\n')
    return train_data,train_labels,valid_data,valid_labels,test_data,test_labels

def blah(args):
    user_labels = [['black', 109], ['orange', 27], ['white', 121]]

    dataloader = fetch()
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split(dataloader, user_labels)

    print("Mean of training data:", np.mean([train_data[i].mean().item() for i in range(len(train_data))]))
    print("Standard Deviation of training data:", np.mean([train_data[i].std().item() for i in range(len(train_data))]))

    training_set = ImageDataset(train_data, train_labels)
    validation_set = ImageDataset(valid_data, valid_labels)
    test_set = ImageDataset(valid_data, valid_labels)
    trainloader = DataLoader(training_set, shuffle=True, batch_size=args.batch_size)
    validloader = DataLoader(validation_set, shuffle=True, batch_size=len(valid_labels))
    testloader = DataLoader(validation_set, shuffle=True, batch_size=len(test_labels))
    classes = np.array([0, 1, 2])
    return

def main(args):
    user_labels = [['black', 102], ['blue',53],['green', 55], ['orange', 38],['red', 55], ['white', 82],['yellow',60]]

    dataloader = fetch()
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split(dataloader, user_labels)

    print("Mean of training data:", np.mean([train_data[i].mean().item() for i in range(len(train_data))]))
    print("Standard Deviation of training data:", np.mean([train_data[i].std().item() for i in range(len(train_data))]))

    training_set = ImageDataset(train_data, train_labels)
    validation_set = ImageDataset(valid_data, valid_labels)
    test_set = ImageDataset(test_data,test_labels)
    trainloader = DataLoader(training_set, shuffle=True, batch_size=args.batch_size)
    validloader = DataLoader(validation_set, shuffle=True, batch_size=len(valid_labels))
    testloader = DataLoader(test_set, shuffle=True, batch_size=len(test_labels))

    net = cnn()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train_acc = []
    train_loss = []
    num_epoch = []
    valid_loss = []
    valid_acc = []
    for epoch in range(args.epochs):
        running_loss = []
        curr_acc = []
        print("\nEpoch:", epoch)

        for i, data in enumerate(trainloader):
            inputs, labels = data
            one_hot_labels = F.one_hot(labels, 10).float()
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()

            stats = [0, 0]
            for i in range(0, outputs.size()[0]):

                if np.argmax(outputs[i].detach().numpy()) == np.argmax(one_hot_labels[i].detach().numpy()):
                    stats[0] += 1
                else:
                    stats[1] += 1

            curr_acc += [stats[0] / outputs.size()[0]]
            running_loss += [loss.item()]
        num_epoch += [epoch]
        train_acc += [np.mean(np.array(curr_acc))]
        train_loss += [np.mean(np.array(running_loss))]

        print('Train Accuracy: ', np.mean(np.array(curr_acc)))
        print('Train Loss: ', np.mean(np.array(running_loss)))

        running_loss = []
        curr_acc = []
        for i, data in enumerate(validloader):
            inputs, labels = data
            one_hot_labels = F.one_hot(labels, 10).float()
            outputs = net(inputs)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()

            stats = [0, 0]
            for i in range(0, outputs.size()[0]):

                if np.argmax(outputs[i].detach().numpy()) == np.argmax(one_hot_labels[i].detach().numpy()):
                    stats[0] += 1
                else:
                    stats[1] += 1
            curr_acc += [stats[0] / len(outputs)]
            running_loss += [loss.item()]
        valid_acc += [np.mean(np.array(curr_acc))]
        valid_loss += [np.mean(np.array(running_loss))]
        print('Validation Accuracy: ', np.mean(np.array(curr_acc)))
        print('Validation Loss: ', np.mean(np.array(running_loss)))

    print('\nFinal Train Accuracy: ', train_acc[len(train_acc) - 1])
    print('Final Train Loss: ', train_loss[len(train_loss) - 1])
    print('\nFinal Validation Accuracy: ', np.mean(np.array(curr_acc)))
    print('Final Validation Loss: ', np.mean(np.array(running_loss)))

    curr_acc = []
    running_loss = []

    for i, data in enumerate(testloader):
        inputs,labels = data
        one_hot_labels = F.one_hot(labels, 10).float()
        outputs = net(inputs)
        loss = criterion(outputs, one_hot_labels)
        loss.backward()
        optimizer.step()

        stats = [0, 0]
        for i in range(0, outputs.size()[0]):
            if np.argmax(outputs[i].detach().numpy()) == np.argmax(one_hot_labels[i].detach().numpy()):
                stats[0] += 1
            else:
                stats[1] += 1
        curr_acc += [stats[0] / len(outputs)]
        running_loss += [loss.item()]
    test_acc = np.mean(np.array(curr_acc))
    test_loss = np.mean(np.array(running_loss))
    print('\nTest Accuracy: ', test_acc)
    print('Test Loss: ', test_loss)

    print("\nFinished Training")

    plt.plot(num_epoch, train_acc, label='Training Accuracy')
    plt.plot(num_epoch, valid_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs. Epoch')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(num_epoch, train_loss, label='Training Loss')
    plt.plot(num_epoch, valid_loss, label='Validation Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.xlabel('# Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='cnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--num_kern', type=int, default=30)
    parser.add_argument('--nf', type=int, default=16)
    parser.add_argument('--type', type=str, default='NA')

    args = parser.parse_args()

    main(args)
