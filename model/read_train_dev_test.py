import numpy as np
import torch

def get_train_dev_test():
    trainX = []
    devX = []
    testX = []
    trainY = []
    devY = []
    testY = []
    with open('data/train.txt') as trainfile:
        for line in trainfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            trainX.append(embed)
            trainY.append(np.argmax(posvec))

    with open('data/dev.txt') as devfile:
        for line in devfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            devX.append(embed)
            devY.append(np.argmax(posvec))

    with open('data/test.txt') as testfile:
        for line in testfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            testX.append(embed)
            testY.append(np.argmax(posvec))

    trainX =  torch.from_numpy(np.asarray(trainX)).float()
    devX = torch.from_numpy(np.asarray(devX)).float()
    testX =  torch.from_numpy(np.asarray(testX)).float()

    trainY =  torch.from_numpy(np.asarray(trainY)).long()
    devY =  torch.from_numpy(np.asarray(devY)).long()
    testY =  torch.from_numpy(np.asarray(testY)).long()
    return trainX, devX, testX, trainY, devY, testY
