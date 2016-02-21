from onestep_dqn_host import Async1StepQLearnerHost
from onestep_dqn_client import Async1StepDQNLearner
from async_target_cnn import AsyncTargetCNN
import numpy as np

def main():
    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    gamename = 'breakout'
    num_actions = 4

    from functools import partial

    cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions)

    learner_count = 16
    learners = list()
    for learner in range(learner_count):
        learners.append(partial(Async1StepDQNLearner, num_actions, cnn.get_parameters()))

    multiprochandler = Async1StepQLearnerHost(cnn, learners, rom, False)

    import time
    st = time.time()
    multiprochandler.run(20)
    multiprochandler.block_until_done()
    et = time.time()
    print('total time', et-st)
    return multiprochandler

# python needs this to run processes
if __name__ == '__main__':
    host = main()