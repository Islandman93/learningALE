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

    learner_count = 8
    learners = list()

    cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions)

    for learner in range(learner_count):
        learners.append(partial(Async1StepDQNLearner, num_actions, cnn.get_parameters()))


    import lasagne
    # print(lasagne.layers.get_all_params(cnn.l_out))
    # exit()
    multiprochandler = Async1StepQLearnerHost(cnn, learners, rom, True)

    import time
    st = time.time()
    multiprochandler.start()
    multiprochandler.block_until_done()
    et = time.time()
    print('total time', et-st)

# python needs this to run processes
if __name__ == '__main__':
    main()