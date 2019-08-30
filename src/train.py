# import
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from os.path import join
import torch.optim as optim
from tqdm import tqdm

# def


def train_loop(dataloader, model, optimizer, loss_function, epochs):
    history = []
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for x, y in dataloader:
            yhat = model(x.view(dataloader.batch_size, -1))
            loss = loss_function(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        history.append(total_loss/dataloader.batch_size)
        print(history[-1])
    return history


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
    datasets_path = '../data'
    batch_size = 32
    lr = 0.05
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
    single_model = MLP(784, 18, 256, 3, 0.5)
    optimizer = optim.Adam(single_model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    #train
    history=train_loop(train_loader,single_model,optimizer,loss_function,epochs)