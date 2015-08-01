__author__ = 'Ben'

from libs.ale_python_interface import ALEInterface
import numpy as np
import time

ale = ALEInterface()

ale.loadROM(b'd:\_code\_reinforcementlearning\\breakout.bin')

(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getLegalActionSet()

frameCount = 0
st = time.time()
for episode in range(10):
    total_reward = 0.0
    while not ale.game_over():
        a = legal_actions[np.random.randint(legal_actions.size)]
        gamescreen = ale.getScreenRGB()
        gamescreen = np.asarray(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], dtype=np.float)
        reward = ale.act(a)
        total_reward += reward
        frameCount += 1
    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    ale.reset_game()
et = time.time()
print(et-st, frameCount/(et-st))

