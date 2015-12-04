__author__ = 'Ben'
import time

import numpy as np

from learningALE.libs.ale_python_interface import ALEInterface

"""
This example is meant for those wanting to play around with GameHandler, or implement their own ALE interface.
For people that want a plug and play interface use learningALE.handlers.GameHandler
"""


# start up the python ale interface
ale = ALEInterface()

# load a rom
ale.loadROM(b'd:\_code\_reinforcementlearning\\breakout.bin')

# screen dimensions and legal actions
(screen_width, screen_height) = ale.getScreenDims()
legal_actions = ale.getLegalActionSet()

frameCount = 0
st = time.time()
for episode in range(1):
    total_reward = 0.0
    while not ale.game_over():
        # get a random action
        a = legal_actions[np.random.randint(legal_actions.size)]

        # get gamescreen and convert to usable format (Height x Width x Channels)
        gamescreen = ale.getScreenRGB()
        gamescreen = np.asarray(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], dtype=np.float)

        ram = ale.getRAM()

        # take the action and get the reward
        reward = ale.act(a)
        total_reward += reward

        frameCount += 1

    print("Episode " + str(episode) + " ended with score: " + str(total_reward))
    # game over man game over, reset
    ale.reset_game()

# end time count and print total time and FPS
et = time.time()
print(et-st, frameCount/(et-st))

