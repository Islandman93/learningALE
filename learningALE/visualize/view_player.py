import time
import matplotlib.pyplot as plt
import numpy as np
from learningALE.handlers.gamehandler import GameHandler
from learningALE.learners.DQN import DQNLearner
from learningALE.tools.life_ram_inds import BREAKOUT


# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
skip_frame = 4
num_actions = 4
learner = DQNLearner(skip_frame, num_actions, load='D:\_code\learningALE\experiments\\reproduction\DQN_Original\dqn675064.0.pkl')

st = time.time()
game_handler = GameHandler(rom, True, learner, skip_frame)
total_reward = game_handler.run_one_game(learner, lives=5, life_ram_ind=BREAKOUT)
et = time.time()
print("Episode ended with score: " + str(total_reward))
print('Total Time:', et - st, 'Frame Count:', game_handler.frameCount, 'FPS:', game_handler.frameCount / (et - st))


