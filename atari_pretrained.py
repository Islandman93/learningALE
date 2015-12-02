__author__ = 'Ben'
import matplotlib.pyplot as plt
import time
import pickle
import lasagne
from nns import *
from handlers.breakouthandler import BreakoutHandler
from functools import partial
import os

rom = b'D:\\_code\\space_invaders.bin'
randVals = (1, 0.1, 500)  # starting at 1 anneal eGreedy policy to 0.1 over 1000 games
breakoutHandler = BreakoutHandler(rom, False, 3, randVals)
cnn = CNN((None, 3, 105, 80), 6, .1)
scoreList = list()
validLossList = list()

st = time.time()
for episode in range(2000):
    # this will run one full game, training every 4 frames
    total_reward = breakoutHandler.runOneGame(cnn.get_output, train=False, negReward=True)
    scoreList.append(total_reward)

    # plot cost and score
    plt.clf()
    plt.subplot(1, 3, 1)
    plt.plot(breakoutHandler.expHandler.costList, '.')
    plt.subplot(1, 3, 2)
    plt.plot(scoreList, '.')
    plt.subplot(1, 3, 3)
    plt.plot(validLossList)
    plt.pause(0.01)

    # uncomment to save network parameters
    if episode % 10 == 0:
        parms = lasagne.layers.get_all_param_values(cnn.l_out)
        # parms2 = lasagne.layers.get_all_param_values(cnn.e_out)
        pickle.dump(parms, open(os.getcwd()+'\saves\cnn_train_target{0}.pkl'.format(episode), 'wb'))

    et = time.time()
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    print('Total Time:', et-st, 'Frame Count:', breakoutHandler.frameCount, 'FPS:', breakoutHandler.frameCount/(et-st))

    #
    # if episode % 2 == 0:
    # for ep in range(50):
    breakoutHandler.expHandler.train_target(50, cnn)
    validLossList.append(breakoutHandler.expHandler.validation(cnn))


plt.ioff()
plt.show()
parms = lasagne.layers.get_all_param_values(cnn.l_out)
with open(os.getcwd() + '\saves\cnn_train_target{0}.pkl'.format(episode+1), 'wb') as fin:
    pickle.dump(parms, fin)
