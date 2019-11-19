import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import cnn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

data_folder = '../necklines'
def fetch():
    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transforms.Compose(
        [transforms.ToTensor()]))
    h={}
    for img in dataset.imgs:
        if img[1] in h:
            h[img[1]] += 1
        else:
            h[img[1]] = 1
    # print(h)
    data_length= len(dataset)
    # print(data_length)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=data_length)
    mean, std = get_mean_std(dataloader)
    # print("Original mean:", mean)
    # print("Original std:", std)

    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std))]))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1)
    mean, std = get_mean_std(dataloader)
    print("Calibrated mean:", mean)
    print("Calibrated std:", std)

    return dataloader, h

def get_mean_std(dataloader):
    mean = []
    std = []
    for i, data in enumerate(dataloader):
        inputs,labels = data
        for j in range(labels.shape[0]):
            mean += [torch.mean(inputs[j])]
            std += [torch.std(inputs[j])]
    return np.mean(mean),np.mean(std)

def get_num(n):
    num_train = int(n * 0.80)
    num_valid = int(n * 0.10)
    num_test = n - num_train - num_valid
    return num_train, num_valid, num_test

def split(dataloader,labels):

    num = []
    stats = []
    for i in range(0,len(labels)):
        num += [[0,0,0]]
        stats += [[0,0,0]]

    for i in range(0,len(labels)):
        num[i][0],num[i][1],num[i][2] = get_num(labels[i])

    num_train = 0
    num_valid = 0
    num_test = 0
    for i in range(0,len(num)):
        num_train += num[i][0]
        num_valid += num[i][1]
        num_test += num[i][2]

    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = [], [], [], [], [], []

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
            print("ERROR: INPUT AND LABELS SIZE MISMATCH")

    return train_data,train_labels,valid_data,valid_labels,test_data,test_labels


def main(args):
    dataloader, h = fetch()
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split(dataloader, h)

    print("Mean of training data:", np.mean([train_data[i].mean().item() for i in range(len(train_data))]))
    print("Standard Deviation of training data:", np.mean([train_data[i].std().item() for i in range(len(train_data))]))

    training_set = ImageDataset(train_data, train_labels)
    validation_set = ImageDataset(valid_data, valid_labels)
    test_set = ImageDataset(test_data,test_labels)
    trainloader = DataLoader(training_set, shuffle=True, batch_size=1)
    validloader = DataLoader(validation_set, shuffle=True, batch_size=1)
    testloader = DataLoader(test_set, shuffle=True, batch_size=1)

    for i, data in enumerate(dataloader):
        inputs, label = data
        print(inputs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=str, default=5)
    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--batch_norm', type=bool, default=True)
    args = parser.parse_args()

    main(args)