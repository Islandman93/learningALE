__author__ = 'Ben'

from libs.ale_python_interface import ALEInterface
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

from handlers.actionhandler import ActionHandler, ActionPolicy
from handlers.experiencehandler import ExperienceHandler
from scipy.misc import imresize
import theano
import lasagne
from nns import CNN

dtype = np.float16
plt.ion()

# set up emulator
ale = ALEInterface(True)
ale.loadROM(b'd:\_code\\breakout.bin')
(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getMinimalActionSet()
lives = 5

# set up vars
skipFrame = 4

actionHandler = ActionHandler(ActionPolicy.eGreedy, (1, 0.1, 1000), legal_actions)
expHandler = ExperienceHandler(32, 0.95, 1000000/skipFrame)

scoreList = list()

cnn = CNN()

frameCount = 0
st = time.time()
for episode in range(2000):
    total_reward = 0.0
    while not ale.game_over():

        # if episode > 0:
        expHandler.train_exp(cnn)

        # get frames
        frames = list()
        reward = 0
        for frame in range(skipFrame):
            gamescreen = ale.getScreenRGB()
            processedImg = np.asarray(imresize(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], 0.5, interp='nearest'), dtype=dtype)/255
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

        actionVect = cnn.get_output(frames.reshape((1, skipFrame, 105, 80)))[0]
        actionHandler.setAction(actionVect)

        total_reward += reward
        frameCount += 1*skipFrame

    ale.reset_game()
    actionHandler.gameOver()
    scoreList.append(total_reward)

    lives = 5

    plt.clf()
    plt.subplot(1,2,1)
    # costArray = np.asarray(costList)
    # std = np.std(costArray)
    plt.plot(expHandler.costList, '.')
    plt.subplot(1,2,2)
    plt.plot(scoreList, '.')
    plt.pause(0.01)

    if episode % 10 == 0:
        parms = lasagne.layers.get_all_param_values(cnn.l_out)
        pickle.dump(parms, open('cnn{0}.pkl'.format(episode), 'wb'))

    print("Episode " + str(episode) + " ended with score: " + str(total_reward))

    et = time.time()
    print('Total Time:', et-st, 'Frame Count:', frameCount, 'FPS:',frameCount/(et-st))

