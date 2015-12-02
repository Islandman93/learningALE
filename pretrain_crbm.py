__author__ = 'Ben'
import matplotlib.pyplot as plt
import time
import pickle
import lasagne
from libs.ale_python_interface import ALEInterface
import numpy as np
import time

from handlers.actionhandler import ActionHandler, ActionPolicy
from handlers.experiencehandler import ExperienceHandler
from scipy.misc import imresize

rom = b'D:\\_code\\space_invaders.bin'
# rom = b'D:\\_code\\breakout.bin'
randVals = (1, 1, 2)
ale = ALEInterface(False)
ale.loadROM(rom)
(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getMinimalActionSet()

# set up vars
skipFrame = 3
dtype = np.float32
actionHandler = ActionHandler(ActionPolicy.eGreedy, randVals, legal_actions)
expHandler = ExperienceHandler(0,0,50000/skipFrame, 6)
frameCount = 0
st = time.time()
for episode in range(1):
    lives = 5
    total_reward = 0.0
    ale.reset_game()
    while not ale.game_over():
        # get frames
        frames = list()
        reward = 0
        for frame in range(skipFrame):
            gamescreen = ale.getScreenRGB()
            processedImg = np.asarray(
                imresize(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[25:-12, :, 0], 0.5, interp='nearest'),
                dtype=dtype)/255
            # print(processedImg.shape)
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

        expHandler.addExperience(frames, actionInd, reward)
        actionHandler.setAction(0)

        total_reward += reward
        frameCount += 1*skipFrame

    # end of game
    actionHandler.gameOver()
    print(episode, frameCount/skipFrame)

trainX = np.asarray(expHandler.states)
print(trainX.shape)
print(trainX.dtype)
from rbms.CRBM import CRBM
from rbms.RBMNode import UnitType
vislayerdimen = trainX.shape[2]
channels = skipFrame
hid1 = 8
hid2 = 8
hidSize = np.asarray([hid1, hid2, channels])

hiddenUnits = 16
rbm = CRBM([vislayerdimen, 80, channels], hidSize, UnitType.nrelu, hiddenUnits, UnitType.nrelu_n, [2, 2], .01,
           sparsityTarget=0, sparsityDecay=0.9, sparsityCost=0,
           hidUnitCap=2, momentum=0.9, patchPct=0.01)
rbm.usePCD = True
st = time.time()
rbm.train(trainX, 100, 1, displayProgress=True, displayMod=100, reset=True)
et = time.time()
dt = et-st
print(dt)
# for c in range(4):
for x in range(16):
    weight = rbm.weights[:, x]
    plt.subplot(8,8,x+1)
    plt.imshow(weight.reshape(hid1, hid2, channels)[:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    # plt.imshow(np.swapaxes(np.swapaxes(weight.reshape(channels, hid1, hid2),0,2),0,1)[:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks()
    plt.yticks()
plt.show()
plt.plot(rbm.histHidAct)
plt.show()
#
# for c in range(channels):
#     plt.subplot(1,skipFrame,c+1)
#     plt.imshow(trainX[50,c], cmap=plt.cm.gray)
# plt.show()

import theano
import theano.tensor as T
from theano.tensor.signal.downsample import max_pool_2d

tIn = T.tensor4()
maxpool = theano.function([tIn], max_pool_2d(tIn, (2, 2)))

#
channels = 16
hid1 = 4
hid2 = 4
hidSize = np.asarray([hid1, hid2, channels])

hiddenUnits = 32
rbm2 = CRBM([40, 37, channels], hidSize, UnitType.nrelu, hiddenUnits, UnitType.nrelu, [2, 2], .001,
           sparsityTarget=0, sparsityDecay=0.9, sparsityCost=0,
           hidUnitCap=2, momentum=0.9, patchPct=0.01)
rbm2.usePCD = False
rbm2.reset_for_training(1)
for ep in range(100):
    for valInd in range(trainX.shape[0]):
        temp = rbm.activate_to_next_layer(trainX[valInd].reshape(1, skipFrame, 86, 80))
        temp = temp.reshape(-1, 79, 73, 16)
        temp = np.swapaxes(np.swapaxes(temp, 1, 3), 2, 3)
        mptemp = maxpool(np.asarray(temp, dtype=np.float32))
        recError = rbm2.run_one_epoch(mptemp)
        if valInd % 100 == 0:
            print(ep, valInd, recError)

for x in range(32):
    weight = rbm2.weights[:, x]
    plt.subplot(6,6,x+1)
    plt.imshow(weight.reshape(hid1, hid2, channels)[:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    # plt.imshow(np.swapaxes(np.swapaxes(weight.reshape(channels, hid1, hid2),0,2),0,1)[:,:,0], cmap=plt.cm.gray, interpolation='nearest')
    plt.xticks()
    plt.yticks()
plt.show()
plt.plot(rbm2.histHidAct)
plt.show()
