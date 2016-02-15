import time
import matplotlib.pyplot as plt
import numpy as np
from learningALE.handlers.gamehandler import GameHandler
from learningALE.learners.DQN import DQNTester


# setup vars
rom = b'D:\\_code\\montezuma_revenge.bin'
gamename = 'montezumas_revenge'
skip_frame = 4
num_actions = 18
learner = DQNTester(skip_frame, num_actions,
                    load='D:\_code\learningALE\experiments\\novelty\dqn2500572.0.pkl',
                    rand_val=0.05)

st = time.time()
game_handler = GameHandler(rom, True, skip_frame, learner)
total_reward = game_handler.run_one_game(learner, clip=False, neg_reward=False)
et = time.time()
print("Episode ended with score: " + str(total_reward))
print('Total Time:', et - st, 'Frame Count:', game_handler.total_frame_count, 'FPS:', game_handler.total_frame_count / (et - st))


