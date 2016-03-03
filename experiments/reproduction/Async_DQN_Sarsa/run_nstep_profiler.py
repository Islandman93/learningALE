from nstep_dqn_client import AsyncNStepDQNLearner

from learningALE.handlers.ale_specific.gamehandler import MinimalGameHandler
from learningALE.learners.async_target_cnn import AsyncTargetCNNNstep

# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
num_actions = 4

cnn = AsyncTargetCNNNstep((None, 4, 84, 84), num_actions, 5)
learner = AsyncNStepDQNLearner(num_actions, cnn.get_parameters(), None)

emulator = MinimalGameHandler(rom, frame_skip=4, show_rom=False)
learner.set_legal_actions(emulator.get_legal_actions())

learner.run_no_pipe(emulator)
learner.run_no_pipe(emulator)
