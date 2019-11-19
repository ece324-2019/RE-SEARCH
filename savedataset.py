from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import cnn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.signal import savgol_filter as sf
import time
import torch.utils.data as data
from sklearn.metrics import confusion_matrix

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

def get_num(n):
    num_train = int(n * 0.80)
    num_valid = int(n * 0.10)
    num_test = n - num_train - num_valid
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

    mean, std = get_mean_std(trainloader)
    print("train mean:", mean, "train std:", std)
    mean, std = get_mean_std(validloader)
    print("valid mean:", mean, "valid std:", std)
    mean, std = get_mean_std(testloader)
    print("test mean:", mean, "test std:", std)

    training_set = ImageDataset(train_data, train_labels)
    validation_set = ImageDataset(valid_data, valid_labels)
    test_set = ImageDataset(test_data,test_labels)
    trainloader = DataLoader(training_set, shuffle=True, batch_size=1)
    validloader = DataLoader(validation_set, shuffle=True, batch_size=1)
    testloader = DataLoader(test_set, shuffle=True, batch_size=1)

    for i in range(0,3):
        if i == 0:
            loader = trainloader
            set = training_set
        elif i == 1:
            loader = validloader
            set = validation_set
        elif i == 2:
            loader = testloader
            set = test_set
        for cnt in range(0,len(set)):
            # get some random training images
            dataiter = iter(loader)
            image, label = dataiter.next()
            path = "./test/" +str(label.detach().numpy()[0]) +'/'
            if os.path.exists(path):
                pass
            else:
                os.mkdir(path)
            torchvision.utils.save_image(image, path + str(cnt) + '.jpg')

if __name__ == '__main__':
    parser.add_argument('--type', type=str, default='colors')
    args = parser.parse_args()
    # print("running model:", args.model, "lr:", args.lr,"batchsize:",args.batch_size,"bn:", args.batch_norm, "dropout:",args.dropout)

    data_folder = '../colors'
    if args.type == 'colors':
        args.class_names = ["black","blue","green","orange","red","white","yellow"]
        args.num_classes = 7
        args.classes = np.array([0,1,2,3,4,5,6])
    elif args.type == 'sleeves':
        args.class_names = ["long","short","sleeveless"]
        args.num_classes = 3
        args.classes = np.array([0, 1, 2, 3, 4, 5, 6])
    elif args.type == 'necklines':
        args.class_names = ["collar","crew","square","turtle","v-neck"]
        args.num_classes = 5
        args.classes = np.array([0, 1, 2, 3, 4, 5, 6])
    elif args.type == 'buttons':
        args.class_names = ["button","no button"]
        args.num_classes = 2
        args.classes = np.array([0,1])

    main(args)