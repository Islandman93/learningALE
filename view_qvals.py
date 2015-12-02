__author__ = 'Ben'
import matplotlib.pyplot as plt
import time
import pickle
import lasagne
from nns import *
from scipy.misc import imresize
from libs.ale_python_interface import ALEInterface
from handlers.actionhandler import ActionHandler, ActionPolicy
from functools import partial
import os

# plt.ion()
skipFrame = 3
cnn = CNN((None, skipFrame, 86, 80), 6, .1, stride=(4,2))
# with open(os.getcwd()+'\saves\\spcinvcnn_stride_dqnbest38.0_0.002.pkl', 'rb') as infile:
with open(os.getcwd()+'\datasets\\spccnn.pkl', 'rb') as infile:
    parms = pickle.load(infile)
    lasagne.layers.set_all_param_values(cnn.l_out, parms)

# rom = b'D:\\_code\\breakout.bin'
rom = b'D:\\_code\\space_invaders.bin'

ale = ALEInterface(True)
ale.loadROM(rom)
(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getMinimalActionSet()
# get labels
labels = ['noop', 'fire', 'up', 'right', 'left', 'down', 'upright', 'upleft', 'downright', 'downleft', 'upfire', 'rightfire', 'leftfire', 'downfire', 'uprightfire'
          , 'upleftfire', 'downrightfire', 'downleftfire']
labels = np.asarray(labels)[legal_actions]

# set up vars
actionHandler = ActionHandler(ActionPolicy.eGreedy, (.1, .1, 2), legal_actions)
rewList = list()
for ep in range(100):
    total_reward = 0.0
    trainCount = 0
    ale.reset_game()
    while not ale.game_over():
        # get frames
        frames = list()
        reward = 0
        for frame in range(skipFrame):
            gamescreen = ale.getScreenRGB()
            processedImg = np.asarray(
                gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[25:-12, :, 0],
                dtype=np.float32)
            processedImg[processedImg > 1] = 255
            processedImg = imresize(processedImg, 0.5, interp='nearest')/255
            frames.append(processedImg)

            performedAction, actionInd = actionHandler.getLastAction()
            rew = ale.act(performedAction)
            if rew > 0:
                rew = 1
            reward += rew
        total_reward += reward
        frames = np.asarray(frames, dtype=np.float32)

        actionVect = cnn.get_output(frames.reshape((1, skipFrame, frames.shape[1], 80)))[0]
        actionHandler.setAction(actionVect)
        # hid1_act = cnn.get_hid1_act(frames.reshape((1, skipFrame, frames.shape[1], 80)))
        # hid2_act = cnn.get_hid2_act(frames.reshape((1, skipFrame, frames.shape[1], 80)))
        # for x in range(hid1_act.shape[1]):
        #     plt.subplot(4,4,x+1)
        #     plt.imshow(hid1_act[0,x], cmap=plt.cm.gray)
        # for x in range(hid2_act.shape[1]):
        #     plt.subplot(6,6,x+1)
        #     plt.imshow(hid2_act[0,x], cmap=plt.cm.gray)
        # plt.show()
        # plt.clf()
        # plt.plot(actionVect)
        # plt.xticks(range(len(labels)), labels)
        # plt.pause(0.001)
    rewList.append(total_reward)
    print(ep, total_reward)


print(np.mean(rewList), np.std(rewList), np.max(rewList), np.min(rewList))
print(np.unique(rewList, return_counts=True))
plt.plot(rewList)
plt.show()