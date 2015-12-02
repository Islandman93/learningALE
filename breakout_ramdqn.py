__author__ = 'Ben'
import matplotlib.pyplot as plt
import time
import pickle
import lasagne
from nns import RAM
from handlers.breakouthandler import BreakoutHandlerRam
from functools import partial

plt.ion()
rom = b'D:\\_code\\space_invaders.bin'
randVals = (1, 0.01, 500)  # starting at 1 anneal eGreedy policy to 0.1 over 1000 games
breakoutHandler = BreakoutHandlerRam(rom, False, 4, randVals)
cnn = RAM()
scoreList = list()
st = time.time()
for episode in range(2000):
    # this will run one full game, training every 4 frames
    total_reward = breakoutHandler.runOneGame(cnn.get_output,
                                              train=True,
                                              trainFn=partial(breakoutHandler.expHandler.train_exp, cnn),
                                              earlyReturn=True)
    scoreList.append(total_reward)

    # plot cost and score
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.plot(breakoutHandler.expHandler.costList, '.')
    plt.subplot(1, 2, 2)
    plt.plot(scoreList, '.')
    plt.pause(0.01)

    # if episode % 2 == 0:
    # for ep in range(50):
    #     breakoutHandler.expHandler.train_exp(cnn)
    et = time.time()
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    print('Total Time:', et-st, 'Frame Count:', breakoutHandler.frameCount, 'FPS:', breakoutHandler.frameCount/(et-st))
