# import
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from os.path import join
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt
from model import *

# def


def train_loop(dataloader, model, optimizer, criterion, epochs):
    train_loader, test_loader = dataloader
    history = []
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = []
        for x, y in train_loader:
            if param['use_cuda']:
                x, y = x.cuda(), y.cuda()
            yhat = model(x.view(-1, 784))
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
    return history


def evaluation(dataloader, model, typ):
    predict = []
    accuracy = []
    model.eval()
    for x, y in dataloader:
        if param['use_cuda']:
            x, y = x.cuda(), y.cuda()
        pred = model(x.view(-1, 784)).cpu().data.numpy()
        if typ == 'single':
            pred = np.argmax(pred, 1)
        if typ == 'multiple':
            pred = np.where(pred > 0.5, 1, 0)
        acc = accuracy_score(y.cpu().data.numpy(), pred)
        predict.append(pred)
        accuracy.append(acc)
    return predict, accuracy


def make_balance_dataloader(data_set, target, transform):
    data, targets = data_set.data.view(-1, 784), data_set.targets
    targets = (targets == target).int()
    ratio = {1: targets.sum().item(), 0: targets.sum().item()}
    data, targets = make_imbalance(data, targets, ratio)
    data_set = TensorsDataset(
        data.reshape(-1, 28, 28), targets.reshape(-1, 1).astype(np.float32), transform)
    return torch.utils.data.DataLoader(dataset=data_set, batch_size=param['batch_size'], shuffle=True, num_workers=param['num_workers'], pin_memory=True)


# class


class TensorsDataset(torch.utils.data.Dataset):

    '''
    A simple loading dataset - loads the tensor that are passed in input. This is the same as
    torch.utils.data.TensorDataset except that you can add transformations to your data and target tensor.
    Target tensor can also be None, in which case it is not returned.
    '''

    def __init__(self, data_tensor, target_tensor=None, transforms=None, target_transforms=None):
        if target_tensor is not None:
            assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

        if transforms is None:
            transforms = []
        if target_transforms is None:
            target_transforms = []

        if not isinstance(transforms, list):
            transforms = [transforms]
        if not isinstance(target_transforms, list):
            target_transforms = [target_transforms]

        self.transforms = transforms
        self.target_transforms = target_transforms

    def __getitem__(self, index):

        data_tensor = self.data_tensor[index]
        for transform in self.transforms:
            data_tensor = transform(data_tensor)

        if self.target_tensor is None:
            return data_tensor

        target_tensor = self.target_tensor[index]
        for transform in self.target_transforms:
            target_tensor = transform(target_tensor)

        return data_tensor, target_tensor

    def __len__(self):
        return self.data_tensor.shape[0]


if __name__ == "__main__":
    # parameters
    param = {'use_cuda': torch.cuda.is_available(),
             'data_path': './data',
             'num_workers': 4,
             'batch_size': 5000,
             'lr': 0.02,
             'epochs': 6}

    # load data
    """range [0, 255] -> [0.0,1.0]"""
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root=param['data_path'], train=True, transform=trans, download=True)
    test_set = datasets.MNIST(
        root=param['data_path'], train=False, transform=trans, download=True)

    for typ in ['single', 'multiple']:
        # create model
        if typ == 'single':
            print('Single model')
            # create dataloader
            train_loader = torch.utils.data.DataLoader(
                dataset=train_set, batch_size=param['batch_size'], shuffle=True, num_workers=param['num_workers'], pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_set, batch_size=param['batch_size'], shuffle=False, num_workers=param['num_workers'], pin_memory=True)

            # create model
            model = MLP(in_dim=784, out_dim=10,
                        hidden_dim=256, n_hidden=1, dropout=0.5)
            optimizer = optim.Adam(model.parameters(), lr=param['lr'])
            criterion = nn.CrossEntropyLoss()
            if param['use_cuda']:
                model = model.cuda()

            # train
            history = train_loop((train_loader, None), model,
                                 optimizer, criterion, param['epochs'])

            # evaluation
            train_predict, train_accuracy = evaluation(
                train_loader, model, typ)
            print('Single model train set accuracy: ', np.mean(train_accuracy))
            test_predict, test_accuracy = evaluation(test_loader, model, typ)
            print('Single model test set accuracy: ', np.mean(test_accuracy))
            print('Finish\n')

        if typ == 'multiple':
            print('Multi model')
            models = []
            histories = []
            for i in range(len(train_set.classes)):
                # create dataloader
                train_loader = make_balance_dataloader(train_set, i, trans)
                test_loader = make_balance_dataloader(test_set, i, trans)

                # create model
                print('Model {}'.format(i))
                model = MLP(in_dim=784, out_dim=1, hidden_dim=256,
                            n_hidden=1, dropout=0.5)
                optimizer = optim.Adam(model.parameters(), lr=param['lr'])
                criterion = nn.BCELoss()
                if param['use_cuda']:
                    model = model.cuda()

                # train
                history = train_loop((train_loader, None), model,
                                     optimizer, criterion, param['epochs'])

                # evaluation
                train_predict, train_accuracy = evaluation(
                    train_loader, model, typ)
                print('Model {} train set accuracy: '.format(
                    i), np.mean(train_accuracy))
                test_predict, test_accuracy = evaluation(
                    test_loader, model, typ)
                print('Model {} test set accuracy: '.format(
                    i), np.mean(test_accuracy))

                # append model and history
                models.append(model)
                histories.append(history)

            # total model evaluation
            train_loader = torch.utils.data.DataLoader(
                dataset=train_set, batch_size=param['batch_size'], shuffle=True, num_workers=param['num_workers'], pin_memory=True)
            test_loader = torch.utils.data.DataLoader(
                dataset=test_set, batch_size=param['batch_size'], shuffle=False, num_workers=param['num_workers'], pin_memory=True)
            total_train_accuracy = []
            for x, y in train_loader:
                if param['use_cuda']:
                    x, y = x.cuda(), y.cuda()
                pred = []
                for i in range(len(train_set.classes)):
                    model = models[i].eval()
                    pred.append(model(x.view(-1, 784)).cpu().data.numpy())
                total_train_accuracy.append(accuracy_score(
                    y.cpu().data.numpy(), np.argmax(pred, 0)))
            total_test_accuracy = []
            for x, y in test_loader:
                if param['use_cuda']:
                    x, y = x.cuda(), y.cuda()
                pred = []
                for i in range(len(test_set.classes)):
                    model = models[i].eval()
                    pred.append(model(x.view(-1, 784)).cpu().data.numpy())
                total_test_accuracy.append(accuracy_score(
                    y.cpu().data.numpy(), np.argmax(pred, 0)))
            print('Total model train set accuracy: {}'.format(
                np.mean(total_train_accuracy)))
            print('Total model test set accuracy: {}'.format(
                np.mean(total_test_accuracy)))
            print('Finish')
