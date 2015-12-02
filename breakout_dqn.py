__author__ = 'Ben'
import matplotlib.pyplot as plt
import time
import pickle
import lasagne
from nns import *
from handlers.breakouthandler import BreakoutHandler
from handlers import ActionHandler, ActionPolicy, ExperienceHandler, TDHandler
from libs.ale_python_interface import ALEInterface
from functools import partial
import os
import copy
# 73 space invaders 57 breakout 5 beamrider
rom = b'D:\\_code\\space_invaders.bin'
gamename = 'spcinv'
skipFrame = 3
numActions = 6
randVals = (.3, 0.1, 250)  # starting at 1 anneal eGreedy policy to 0.1 over 1000 games
expHandler = ExperienceHandler(32, 0.95, 1000000/skipFrame, numActions)
breakoutHandler = BreakoutHandler(rom, False, randVals, expHandler, skipFrame)
cnn = CNN((None, skipFrame, 86, 80), numActions, .1)
with open(os.getcwd()+'\datasets\\cnn.pkl', 'rb') as infile:
    parms = pickle.load(infile)
    lasagne.layers.set_all_param_values(cnn.l_out, parms)
scoreList = list()
validLossList = list()
bestTotReward = -np.inf

st = time.time()
for episode in range(2000):
    # this will run one full game, training every 4 frames
    # total_reward = breakoutHandler.runOneGame(expHandler.addExperience, cnn.get_output, 3, 73, train=True,
    #                                           trainFn=partial(breakoutHandler.expHandler.train_exp, cnn), negReward=True)
    total_reward = breakoutHandler.runOneGame(expHandler.addExperience, cnn.get_output, 3, 73, train=False, negReward=True)
    scoreList.append(total_reward)

    # if this is the best score save it as such
    if total_reward >= bestTotReward:
        parms = lasagne.layers.get_all_param_values(cnn.l_out)
        #parms2 = lasagne.layers.get_all_param_values(cnn.e_out)
        pickle.dump(parms, open(os.getcwd()+'\saves\\'+gamename+'cnn_stride_dqnbest{0}_0.002.pkl'.format(total_reward), 'wb'))
        bestTotReward = total_reward
    # plot cost and score
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(expHandler.costList, '.')
    plt.subplot(1, 2, 2)
    sl = np.asarray(scoreList)
    plt.plot(sl, '.')
    # plt.subplot(1, 3, 3)
    # plt.plot(validLossList)
    plt.pause(0.01)

    # uncomment to save network parameters
    if episode % 10 == 0:
        parms = lasagne.layers.get_all_param_values(cnn.l_out)
        #parms2 = lasagne.layers.get_all_param_values(cnn.e_out)
        pickle.dump(parms, open(os.getcwd()+'\saves\\'+gamename+'cnn_stride_dqn{0}_0.002.pkl'.format(episode), 'wb'))

    et = time.time()
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    print('Total Time:', et-st, 'Frame Count:', breakoutHandler.frameCount, 'FPS:', breakoutHandler.frameCount/(et-st))

    #
    # if episode % 2 == 0:
    ttt = 0
    vtt = 0
    trainep = episode
    if trainep > 50:
        trainep = 50
    for ep in range(trainep):
        tst = time.time()
        breakoutHandler.expHandler.train_exp(cnn)
        tet = time.time()
        ttt += tet-tst
    vst = time.time()
    # validLossList.append(breakoutHandler.expHandler.validation(cnn))
    vet = time.time()
    vtt += vet-vst
    print(ttt,vtt)

plt.ioff()
plt.show()
parms = lasagne.layers.get_all_param_values(cnn.l_out)
with open(os.getcwd() + '\saves\\'+gamename+'cnn_stride_dqn{0}_0.002.pkl'.format(episode+1), 'wb') as fin:
    pickle.dump(parms, fin)
