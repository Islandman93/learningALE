import os
import numpy as np
from learningALE.visualize.evaluate_player import evaluate_player
from learningALE.learners.DQN import DQNTester
from learningALE.handlers.gamehandler import GameHandler

os.chdir('D:\_code\learningALE\experiments\\reproduction\DQN_Original')

epoch_files = list()
for file in os.listdir(os.getcwd()):
    if 'dqn' in file and '.pkl' in file and 'best' not in file:
        epoch_files.append(file)

epoch_dqn = list()
for dqn in epoch_files:
    epoch = dqn.replace('dqn', '')
    epoch = epoch.replace('.pkl', '')
    epoch = round(float(epoch) / 50000, 1)
    epoch_dqn.append([epoch, dqn])

epoch_dqn.sort()

# setup vars
rom = b'D:\\_code\\breakout.bin'
skip_frame = 4
num_actions = 4
game_handler = GameHandler(rom, False, skip_frame)

# now test to plot score
num_to_run = 10
means = list()
q_vals = list()
for ep, file in epoch_dqn:
    player_file = os.getcwd() + '\\' + file
    learner = DQNTester(skip_frame, num_actions, load=player_file)
    game_handler.set_legal_actions(learner)

    rews, times, frames = evaluate_player(learner, game_handler, num_to_run)
    means.append(np.mean(rews))
    qval_arr = np.asarray(learner.q_vals)
    qval_arr[np.isinf(qval_arr)] = np.nan
    mean_qvals = np.nanmean(qval_arr, axis=0)
    q_vals.append(mean_qvals)
    print(ep, means[-1], mean_qvals, file, np.sum(times))

import matplotlib.pyplot as plt
plt.plot(means)
plt.xticks(np.arange(len(means)), np.asarray(epoch_dqn)[:, 0], rotation='vertical')
plt.show()
