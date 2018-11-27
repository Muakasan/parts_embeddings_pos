#https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py
from read_train_dev_test import get_train_dev_test
from model import Net
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch

def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        '''
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    trainX, devX, testX, trainY, devY, testY = get_train_dev_test()

    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=128, shuffle=True)
    dev_loader = DataLoader(TensorDataset(devX, devY))

    model = Net(False)
    optimizer = optim.SGD(model.parameters(), lr=.01) #Add momentum?

    '''
    for epoch in range(1, 100+1):
        train(model, train_loader, optimizer, epoch)
        test(model, dev_loader)
    '''
if __name__ == '__main__':
    main()