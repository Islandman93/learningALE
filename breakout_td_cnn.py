__author__ = 'Ben'

from libs.ale_python_interface import ALEInterface
import matplotlib.pyplot as plt
import numpy as np
import time
from handlers import *
from scipy.misc import imresize
import theano
from nns import CNN

plt.ion()
ale = ALEInterface(True)

ale.loadROM(b'd:\_code\\breakout.bin')

(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getMinimalActionSet()

experienceReplay = True
expEpoch = 200
expMiniBatch = 10
discount = 1
histNumSeconds = 2
skipFrame = 4

actionHandler = ActionHandler(ActionPolicy.eGreedy, (1, 0.1, 200), legal_actions)
historyHandler = HistoryHandler(60*histNumSeconds)
expHandler = ExperienceHandler(1500)

costList = list()
scoreList = list()
lives = 5

cnn = CNN()

def train_reward(reward):
    cost = 0
    currentReward = reward

    lastKeys, lastActions = historyHandler.getKeysActions()
    lastKeys = np.asarray(lastKeys, dtype=theano.config.floatX)
    lastActions = np.asarray(lastActions, dtype=int)
    rewVec = list()
    for x in range(len(lastKeys)):
        rewVec.append(currentReward)
        currentReward *= discount
    rewVec = np.asarray(rewVec, dtype=theano.config.floatX)
    rp = np.random.permutation(len(lastKeys))
    for ind in range(int(len(rp)/expMiniBatch)):
        choice = rp[ind*expMiniBatch:(ind*expMiniBatch)+expMiniBatch]
        key = lastKeys[choice]
        action = lastActions[choice]
        actionArange = [np.arange(expMiniBatch), action]
        rewVal = np.zeros((expMiniBatch, actionHandler.numActions), dtype=theano.config.floatX)
        rewVal[actionArange[0], actionArange[1]] = rewVec[choice]
        mask = np.zeros(rewVal.shape, dtype=theano.config.floatX)
        nonZero = np.where(rewVal != 0)
        mask[nonZero[0], nonZero[1]] = 1
        cost += cnn.train(key, rewVal, mask)

        expHandler.addExperience(key, rewVal)
    # print(cost)
    # costList.append(cost)
    historyHandler.reset()


def train_exp():
    cost = 0
    for x in range(expEpoch):
        keys, actions, isgood = expHandler.getRandomExperience()
        if isgood:
            mask = np.zeros(actions.shape, dtype=theano.config.floatX)
            nonZero = np.where(actions != 0)
            mask[nonZero[0], nonZero[1]] = 1
            cost = cnn.train(keys, actions, mask)

    print(cost/expEpoch)
    costList.append(cost/expEpoch)

frameCount = 0
st = time.time()
for episode in range(200):
    total_reward = 0.0
    while not ale.game_over():

        frames = list()
        reward = 0
        for frame in range(skipFrame):
            gamescreen = ale.getScreenRGB()
            processedImg = np.asarray(imresize(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], 0.5, interp='nearest'), dtype=theano.config.floatX)/255
            frames.append(processedImg)

            performedAction, actionInd = actionHandler.getLastAction()
            rew = ale.act(performedAction)
            if rew > 0:
                reward += 1

            ram = ale.getRAM()
            if ram[57] != lives:
                reward -= 1
                lives = ram[57]

        frames = np.asarray(frames)

        # add relevant information to history handler
        historyHandler.add(frames, actionInd)

        if reward != 0:
            train_reward(reward)

        actionVect = cnn.get_output(frames.reshape((1, skipFrame, 105, 80)))[0]

        actionHandler.setAction(actionVect)

        total_reward += reward
        frameCount += 1*skipFrame

    ale.reset_game()
    actionHandler.gameOver()
    historyHandler.addPScore(total_reward)
    lives = 5
    historyHandler.reset()

    # train_exp()
    #
    # plt.clf()
    # plt.subplot(1,2,1)
    # plt.plot(costList)
    # plt.subplot(1,2,2)
    # plt.plot(historyHandler.pScoreList)
    # plt.pause(0.1)
    #
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))

# episodes over
et = time.time()
print(et-st, frameCount/(et-st))

