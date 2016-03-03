from nstep_a3c_client import AsyncNStepA3CLearner
from learningALE.handlers.ale_specific.gamehandler import MinimalGameHandler
from learningALE.learners.async_a3c_cnn import AsyncA3CCNN

# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
num_actions = 4

cnn = AsyncA3CCNN((None, 4, 84, 84), num_actions, 5)
learner = AsyncNStepA3CLearner(num_actions, cnn.get_parameters(), None)

emulator = MinimalGameHandler(rom, frame_skip=4, show_rom=True)
learner.set_legal_actions(emulator.get_legal_actions())

import copy
loss_list = list()
for ep in range(10):
    learner.run_no_pipe(emulator)
    loss_list += copy.deepcopy(learner.loss_list)

import matplotlib.pyplot as plt
import numpy as np
a = np.asarray(loss_list)
plt.subplot(1,2,1)
plt.plot(a[:, 0])
plt.subplot(1,2,2)
plt.plot(a[:, 1])
plt.show()