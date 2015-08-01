__author__ = 'Ben'
import numpy as np


class ActionHandler:
    def __init__(self, actionPolicy, randVal, actions):
        self.actionPolicy = actionPolicy
        self.actions = actions
        self.lastAction = 0
        self.lastActionInd = 0

        self.numActions = len(actions)
        self.randVal = randVal[0]
        self.lowestRandVal = randVal[1]
        lin = np.linspace(randVal[0], randVal[1], randVal[2])
        self.diff = lin[0] - lin[1]

        self.countRand = 0
        self.actionCount = 0

    def getAction(self, action, random=True):
        if random:
            if self.actionPolicy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action
                if np.random.uniform(0, 1) <= self.randVal:
                    action = np.random.randint(self.numActions)
                    self.countRand += 1
                else:
                    action = np.where(action == np.max(action))[0][0]
            elif self.actionPolicy == ActionPolicy.randVals:
                assert np.iterable(action)
                action += np.random.randn(self.numActions) * self.randVal
                action = np.where(action == np.max(action))[0][0]
        self.actionCount += 1

        return self.actions[action], action

    def setAction(self, action, random=True):
        self.lastAction, self.lastActionInd = self.getAction(action, random)

    def getLastAction(self):
        return self.lastAction, self.lastActionInd

    def gameOver(self):
        self.randVal -= self.diff
        if self.randVal < self.lowestRandVal:
            self.randVal = self.lowestRandVal

from enum import Enum
class ActionPolicy(Enum):
    eGreedy = 1
    randVals = 2