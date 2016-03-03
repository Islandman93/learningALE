import time

import matplotlib.pyplot as plt
import numpy as np
from NoveltyLearner import NoveltyLearner

from learningALE.handlers.ale_specific.gamehandler import GameHandler

# setup vars
rom = b'D:\\_code\\montezuma_revenge.bin'
gamename = 'breakout'
skip_frame = 4
num_actions = 18
learner = NoveltyLearner(skip_frame, num_actions)


def main(epochs):
    ep_count = 0
    game_handler = GameHandler(rom, True, skip_frame, learner)
    scoreList = list()
    bestTotReward = -np.inf
    plt.ion()
    st = time.time()
    while game_handler.frameCount/skip_frame < epochs*50000:
        total_reward = game_handler.run_one_game(learner, neg_reward=True, early_return=True)
        scoreList.append(total_reward)

        learner.game_over()

        # if this is the best score save it as such
        if total_reward >= bestTotReward:
            learner.save('dqnbest{0}.pkl'.format(total_reward))
            bestTotReward = total_reward

        # save params every 25000 updates (half an epoch)
        if (game_handler.frameCount/skip_frame) >= ep_count * 50000:
            ep_count += 0.5
            # plot cost and score
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(learner.get_cost_list(), '.')
            plt.subplot(1, 2, 2)
            sl = np.asarray(scoreList)
            plt.plot(sl, '.')
            plt.pause(0.0001)
            learner.save('dqn{0}.pkl'.format(game_handler.frameCount/skip_frame))

        et = time.time()
        # print("Episode ended with score: " + str(total_reward))
        print('Novel frames', len(learner.frame_table), 'Frame Count:', game_handler.frameCount, 'pct novel', len(learner.frame_table)/game_handler.frameCount)

    plt.ioff()
    plt.show()

    # final save
    learner.save('dqn{0}.pkl'.format(game_handler.frameCount/skip_frame))

if __name__ == '__main__':
    main(50)


