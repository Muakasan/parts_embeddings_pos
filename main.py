#https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py
from model.read_train_dev_test import get_train_dev_test
from model.model import Net
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import torch

EMBED_DIM = 300
NUM_PARTITIONS = 2

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

def run_model(trainX, trainY, testX, testY, part):
    train_loader = DataLoader(TensorDataset(trainX, trainY), batch_size=128, shuffle=True)
    test_loader = DataLoader(TensorDataset(testX, testY))
    model = None
    if part:
        model = Net(part=True, embed_dim=EMBED_DIM, num_splits=NUM_PARTITIONS)
    else:
        model = Net(part=False, embed_dim=EMBED_DIM, num_splits=NUM_PARTITIONS)
    optimizer = optim.SGD(model.parameters(), lr=.01) #Add momentum?
    for epoch in range(50):
        train(model, train_loader, optimizer, epoch)
    test(model, test_loader)

def main():
    trainX, devX, testX, trainY, devY, testY = get_train_dev_test()
    trainX, devX, testX = torch.from_numpy(trainX).float(), torch.from_numpy(devX).float(), torch.from_numpy(testX).float()
    trainY, devY, testY = torch.from_numpy(trainY).long(), torch.from_numpy(devY).long(), torch.from_numpy(testY).long()

    print("Full:")
    run_model(trainX, trainY, testX, testY, part=False)
    for i in range(NUM_PARTITIONS):
        print("Part {0}:".format(i+1))
        psize = int(EMBED_DIM/NUM_PARTITIONS)
        run_model(trainX[:, i*psize:(i+1)*psize], trainY, testX[:, i*psize:(i+1)*psize], testY, True)


if __name__ == '__main__':
    main()
