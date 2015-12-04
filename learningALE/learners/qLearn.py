import pickle
import numpy as np

class QLearn():
    def __init__(self, numVals):
        self.qMap = dict()
        self.numVals = numVals

    def addKey(self, key, value=None):
        if key not in self.qMap:
            if value is None:
                value = np.zeros(self.numVals)
            self.qMap[key] = value

        return key

    def save(self, filename, prune=False, pruned=False):
        try:
            with open(filename, 'wb') as output:
                pickle.dump(self.qMap, output, pickle.HIGHEST_PROTOCOL)
            return 1
        except MemoryError:
            if prune:
                print("Pruning qMap and resaving")
                self.prune()
                self.save(filename, prune=False, pruned=True)
            if pruned:
                print("QMap is already pruned but still can't save, sorry")
                return 0

    def load(self, filename):
        with open(filename, 'rb') as input:
            self.qMap = pickle.load(input)

    def prune(self):
        qlist = list(self.qMap.items())
        qkeys = list(self.qMap.keys())

        qdellist = list()
        for ind in range(len(qlist)):
            if np.max(np.abs(qlist[ind][1])) == 0:
                qdellist.append(qkeys[ind])

        for delKey in qdellist:
            del self.qMap[delKey]

    def view(self, numSamples):
        import numpy as np
        import matplotlib.pyplot as plt

        qList = list(self.qMap.items())

        qValList = list()
        for ind in range(len(qList)):
            qValList.append(qList[ind][1])

        qValList = np.asarray(qValList)
        qValListSize = qValList.shape[1]

        for action in range(qValListSize):
            plt.subplot(1, qValListSize, action+1)
            plt.hist(qValList[:, action], bins=50)
        plt.show()

        for x in range(numSamples):
            randint = np.random.randint(len(qList))
            qImg = np.asarray(qList[randint][0])
            qVal = np.asarray(qList[randint][1])
            plt.subplot(1, 2, 1)
            plt.imshow(qImg, cmap=plt.cm.gray, interpolation='nearest')
            plt.subplot(1, 2, 2)
            plt.plot(qVal)
            plt.show()

    def getRandomExperience(self):
        qList = list(self.qMap.items())
        lenQ = len(qList)
        if lenQ > 0:
            randint = np.random.randint(lenQ)
            qImg = qList[randint][0]
            qVal = qList[randint][1]
            while np.all(qVal == np.zeros(self.numVals)):
                randint = np.random.randint(len(qList))
                qImg = qList[randint][0]
                qVal = qList[randint][1]
            return qImg, qVal, 1
        else:
            return None, None, 0

    def getAction(self, key):
        return self.qMap[key]

    def getSize(self):
        import sys
        return sys.getsizeof(self.qMap)

    # neither of these match the actual Q update which is Q[i,a] = Q[i,a] - alpha*Q[i,a] + alpha * [discount * max(Q(i+1)]
    def addReward(self, key, action, reward):
        if key in self.qMap:
            self.qMap[key][action] += reward
        else:
            vals = np.zeros(self.numVals)
            vals[action] = reward
            self.qMap[key] = vals

    def setReward(self, key, action, reward):
        if key in self.qMap:
            self.qMap[key][action] = reward
        else:
            vals = np.zeros(self.numVals)
            vals[action] = reward
            self.qMap[key] = vals


def main():
    q = QLearn(18)
    q.load('libs/qLearn.pkl')
    q.view(10)
    return

if __name__ == "__main__":
    main()