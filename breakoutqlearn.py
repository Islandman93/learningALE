__author__ = 'Ben'

from libs.ale_python_interface import ALEInterface
import numpy as np
import time
from handlers import *
from qLearn import QLearn
from scipy.misc import imresize
from cnn import CNN
import theano

ale = ALEInterface()

ale.loadROM(b'd:\_code\\breakout.bin')

(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getMinimalActionSet()

experienceReplay = True
expEpoch = 10
expMiniBatch = 10
discount = 0.95
learningRate = 0.01
momentum = 0.9
histNumSeconds = 2
qLearn = QLearn(len(legal_actions))

actionHandler = ActionHandler(ActionPolicy.eGreedy, (1, 0.1, 100), legal_actions)
historyHandler = HistoryHandler(60*histNumSeconds)

lives = 5

cnn = CNN()

def train_reward(reward):
    lastKeys, lastActions = historyHandler.getKeysActions()
    lastKeys = lastKeys
    lastActions = np.asarray(lastActions, dtype=int)

    for ind in range(len(lastKeys)):
        key = lastKeys[ind]
        action = lastActions[ind]

        if ind == 0:
            qLearn.setReward(key, action, reward)
        else:
            actions = qLearn.getAction(lKey)
            mQ = np.max(actions[np.where(actions!=0)[0]])
            rew = discount * mQ
            qLearn.setReward(key, action, rew)

        lKey = key

    historyHandler.reset()


def train_exp():
    for x in range(expEpoch):
        keys, actions, isgood = qLearn.getRandomExperience()
        if isgood:
            cnn.train(np.asarray(keys, dtype=theano.config.floatX).reshape((1, 1, 105, 80)), np.asarray(actions, dtype=theano.config.floatX).reshape((1, 3)))

costList = list()
scoreList = list()

frameCount = 0
st = time.time()
for episode in range(100):
    total_reward = 0.0
    while not ale.game_over():
        gamescreen = ale.getScreenRGB()
        resizedImg = imresize(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], 0.5, interp='nearest')
        processedImg = np.around(np.asarray(resizedImg, dtype=theano.config.floatX)/255, decimals=3)

        # this will add the key and initialize to 0, unless key exists
        qKey = tuple(processedImg.flatten())
        qLearn.addKey(qKey)

        actionVect = cnn.get_output(processedImg.reshape((1, 1, 105, 80)))[0]

        performedAction, actionInd = actionHandler.getAction(actionVect)

        # add relevant information to history handler
        historyHandler.add(qKey, actionInd)
        reward = ale.act(performedAction)

        ram = ale.getRAM()
        if ram[57] != lives:
            reward = -1
            lives = ram[57]

        if reward != 0:
            train_reward(reward)

        total_reward += reward
        frameCount += 1

    ale.reset_game()

    # Game over
    train_exp()
    qLearn.prune()
    actionHandler.gameOver()

    historyHandler.addPScore(total_reward)
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))

# episodes over
et = time.time()
print(et-st, frameCount/(et-st))
qLearn.save('qLearn.pkl', prune=True)

