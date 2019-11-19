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

torch.manual_seed(0)
""" Type of classifier """
""" 1) args.type = color """
""" 2) args.type = buttons """
""" 3) args.type = neck """
""" 4) args.type = sleeve """

""" Testing case: black, white, orange """
""" import photos with correct labels """

data_folder = '../necklines'


def get_mean_std(dataloader):
    mean = []
    std = []
    for i, data in enumerate(dataloader):
        inputs, labels = data
        for j in range(labels.shape[0]):
            mean += [torch.mean(inputs[j])]
            std += [torch.std(inputs[j])]
    return np.mean(mean), np.mean(std)


def fetch():
    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transforms.Compose(
        [transforms.ToTensor()]))
    h = {}
    for img in dataset.imgs:
        if img[1] in h:
            h[img[1]] += 1
        else:
            h[img[1]] = 1
    # print(h)
    data_length = len(dataset)
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
        return self.X[index], self.y[index]


def split(dataloader, labels):
    # testing case:
    # labels should be type [['white',121],['black',109],['orange',27]]

    num = []
    stats = []
    for i in range(0, len(labels)):
        num += [[0, 0, 0]]
        stats += [[0, 0, 0]]

    for i in range(0, len(labels)):
        num[i][0], num[i][1], num[i][2] = get_num(labels[i])

    num_train = 0
    num_valid = 0
    num_test = 0
    for i in range(0, len(num)):
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

    return train_data, train_labels, valid_data, valid_labels, test_data, test_labels


def eval(model, loader, loss_fnc, optimizer=None, train=False):
    running_loss = []
    curr_acc = []
    for i, data in enumerate(loader):
        inputs, labels = data
        one_hot_labels = F.one_hot(labels, args.num_classes).float()
        if args.loss_function == "MSE":
            labels = one_hot_labels
        if train:
            model.train()
            optimizer.zero_grad()
        else:
            model.eval()
        outputs = model(inputs, args.batch_norm)
        loss = loss_fnc(outputs, labels)
        if train:
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
    acc = np.mean(np.array(curr_acc))
    loss = np.mean(np.array(running_loss))
    return acc, loss


def main(args):
    dataloader, h = fetch()
    train_data, train_labels, valid_data, valid_labels, test_data, test_labels = split(dataloader, h)

    print("Mean of training data:", np.mean([train_data[i].mean().item() for i in range(len(train_data))]))
    print("Standard Deviation of training data:", np.mean([train_data[i].std().item() for i in range(len(train_data))]))

    training_set = ImageDataset(train_data, train_labels)
    validation_set = ImageDataset(valid_data, valid_labels)
    test_set = ImageDataset(test_data, test_labels)
    trainloader = DataLoader(training_set, shuffle=True, batch_size=args.batch_size)
    validloader = DataLoader(validation_set, shuffle=True, batch_size=len(valid_labels))
    testloader = DataLoader(test_set, shuffle=True, batch_size=len(test_labels))

    model = cnn(args.num_classes)
    if args.loss_function == "CE":
        loss_fnc = nn.CrossEntropyLoss()
    elif args.loss_function == "MSE":
        loss_fnc = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

    train_acc, train_loss, valid_loss, valid_acc = [], [], [], []
    for epoch in range(args.epochs):
        t_acc, t_loss = eval(model=model, loss_fnc=loss_fnc, optimizer=optimizer, loader=trainloader, train=True)
        train_acc += [t_acc]
        train_loss += [t_loss]
        v_acc, v_loss = eval(model=model, loss_fnc=loss_fnc, loader=validloader)
        valid_loss += [v_loss]
        valid_acc += [v_acc]
        print("Epoch", epoch, "Train Acc:", round(t_acc, 3), "Valid Acc:", round(v_acc, 3), "Train Loss",
              round(t_loss, 3), "Valid Loss:", round(v_loss, 3))

    print('\nFinal Train Accuracy: ', train_acc[-1])
    print('Final Train Loss: ', train_loss[-1])
    print('\nFinal Validation Accuracy: ', v_acc)
    print('Final Validation Loss: ', v_loss)

    test_acc, test_loss = eval(model=model, loss_fnc=loss_fnc, loader=testloader)
    print('\nTest Accuracy: ', test_acc)
    print('Test Loss: ', test_loss)

    print("\nFinished Training")

    plt.plot(range(len(train_acc)), train_acc, label='Training Accuracy')
    plt.plot(range(len(valid_acc)), valid_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs. Epoch')
    plt.xlabel('# Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(valid_loss)), valid_loss, label='Validation Loss')
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
    parser.add_argument('--num_classes', type=str, default=5)
    parser.add_argument('--loss_function', type=str, default='CE')
    parser.add_argument('--batch_norm', type=bool, default=True)
    args = parser.parse_args()

    main(args)