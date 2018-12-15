#Aidan San
import numpy as np

def get_train_dev_test(location='data/'):
    trainX = []
    devX = []
    testX = []
    trainY = []
    devY = []
    testY = []
    with open(location + 'train.txt') as trainfile:
        for line in trainfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            trainX.append(embed)
            trainY.append(np.argmax(posvec))

    with open(location + 'dev.txt') as devfile:
        for line in devfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            devX.append(embed)
            devY.append(np.argmax(posvec))

    with open(location + 'test.txt') as testfile:
        for line in testfile:
            word, embed, posvec = line.split('\t')
            embed = np.fromstring(embed, sep=',')
            posvec = np.fromstring(posvec, sep=',')
            testX.append(embed)
            testY.append(np.argmax(posvec))

    trainX = np.asarray(trainX)
    devX = np.asarray(devX)
    testX =  np.asarray(testX)

    trainY =  np.asarray(trainY)
    devY =  np.asarray(devY)
    testY =  np.asarray(testY)
    return trainX, devX, testX, trainY, devY, testY
