from onestep_dqn_client import Async1StepDQNLearner

from learningALE.handlers.ale_specific.gamehandler import MinimalGameHandler
from learningALE.learners.async_target_cnn import AsyncTargetCNN

# setup vars
rom = b'D:\\_code\\breakout.bin'
gamename = 'breakout'
num_actions = 4
discount = 0.95

cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions, discount)
learner = Async1StepDQNLearner(num_actions, cnn.get_parameters(), None)

emulator = MinimalGameHandler(rom, frame_skip=4, show_rom=False)
learner.set_legal_actions(emulator.get_legal_actions())

learner.profile(emulator)
