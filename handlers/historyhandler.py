__author__ = 'Ben'
import numpy as np


class HistoryHandler():
    def __init__(self, lastRange):
        self.lastKeys = list()
        self.lastActions = list()
        self.pScoreList = list()
        self.lastRange = lastRange

    def add(self, qKey, actionPerformed):
        self.lastKeys.append(qKey)
        self.lastActions.append(actionPerformed)
        
        self.deleteOverLength()
        
    def deleteOverLength(self):
        if len(self.lastActions) > self.lastRange:
            del self.lastActions[0]
        if len(self.lastKeys) > self.lastRange:
            del self.lastKeys[0]
        # pass

    def addPScore(self, pScore):
        self.pScoreList.append(pScore)

    def reset(self):
        self.lastKeys = list()
        self.lastActions = list()

    def getKeysActionStripped(self):

        # get the sum of the last image
        sumLast = np.sum(self.lastKeys[-1])
        # if sumLast > 1m

        # find where the image has a sum greater than the last image
        for x in reversed(range(len(self.lastKeys))):
            if np.sum(self.lastKeys[x]) != sumLast:
                break

        # now we have indicator return all images before this (inclusive)
        return self.lastKeys[0:x], self.lastActions[0:x]

    def getKeysActions(self):
        return self.lastKeys, self.lastActions




