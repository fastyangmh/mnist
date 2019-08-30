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

# def


def train_loop(dataloader, model, optimizer, criterion, epochs):
    history = []
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y in dataloader:
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            yhat = model(x.view(dataloader.batch_size, -1))
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss/dataloader.batch_size)
        print('\n', round(history[-1], 4))
    return history


def evaluation(dataloader, model):
    predict = []
    accuracy = []
    model.eval()
    for x, y in dataloader:
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        pred = model(x).cpu().data.numpy()
        pred = np.argmax(pred, 1)
        acc = accuracy(y.cpu().data.numpy(), pred)
        predict.append(pred)
        accuracy.append(acc)
    return predict, accuracy


# class


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout):
        super(MLP, self).__init__()
        in_sizes = [in_dim]+[hidden_dim]*(n_hidden-1)
        out_sizes = [hidden_dim]*n_hidden
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size) for (
            in_size, out_size) in zip(in_sizes, out_sizes)])
        self.last_layer = nn.Linear(hidden_dim, out_dim)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.leakyrelu(layer(x)))
        x = self.sigmoid(self.last_layer(x))
        return x


if __name__ == "__main__":
    # parameters
    use_cuda = torch.cuda.is_available()
    datasets_path = '../data'
    batch_size = 1000
    lr = 0.001
    epochs = 20

    # load data
    # range [0, 255] -> [0.0,1.0]
    trans = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(
        root=datasets_path, train=True, transform=trans, download=True)
    test_set = datasets.MNIST(
        root=datasets_path, train=False, transform=trans, download=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False)

    # create model
    single_model = MLP(784, 10, 256, 3, 0.5)
    optimizer = optim.Adam(single_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    if use_cuda:
        single_model = single_model.cuda()

    # train
    history = train_loop(train_loader, single_model,
                         optimizer, criterion, epochs)
    print(history)

    # evaluation
    predict, accuracy = evaluation(test_loader, model)
