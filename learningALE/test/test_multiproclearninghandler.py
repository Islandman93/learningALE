from learningALE.handlers.multiproclearninghandler import MultiprocLearningHandler
from learningALE.learners.DQN import DQNLearner
import numpy as np


class randomAgent():
    def __init__(self):
        self.actions = None

    def frames_processed(self, frames, action_performed, reward):
        pass

    def get_game_action(self):
        rand_action = np.random.randint(0, len(self.actions))
        return self.actions[rand_action]

    def game_over(self):
        pass

    def set_legal_actions(self, legal_actions):
        self.actions = legal_actions

def main():
    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    gamename = 'breakout'
    skip_frame = 4

    from functools import partial

    learner_count = 4
    learners = list()

    for learner in range(learner_count):
        learners.append(partial(DQNLearner, skip_frame, 4))

    multiprochandler = MultiprocLearningHandler(learners, rom, False, skip_frame)

    import time
    st = time.time()
    multiprochandler.run_all()
    multiprochandler.block_until_done()
    et = time.time()
    print('total time', et-st)

# python needs this to run processes
if __name__ == '__main__':
    main()