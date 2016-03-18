from learningALE.handlers.async.AsyncHost import AsyncLearnerHost
from learningALE.handlers.async.AsyncClient import AsyncClientProcess
from nstep_dqn_client import AsyncNStepDQNLearner
from learningALE.learners.async_target_cnn import AsyncTargetCNNNstep


def main():
    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    gamename = 'breakout'
    num_actions = 4
    discount = 0.95
    learner_count = 8
    epochs = 15
    status_interval = 0.1

    from functools import partial

    cnn = AsyncTargetCNNNstep((None, 4, 84, 84), num_actions, 5)

    learners = list()
    for learner in range(learner_count):
        learner_process = (partial(AsyncNStepDQNLearner, num_actions, cnn.get_parameters(), discount=discount), AsyncClientProcess)
        learners.append(learner_process)

    host = AsyncLearnerHost(cnn, learners, rom)

    import time
    st = time.time()
    host.run(epochs, save_interval=status_interval, show_status=True)
    host.block_until_done()
    et = time.time()
    print('total time', et-st)
    return host

# python needs this to run processes
if __name__ == '__main__':
    try:
        host = main()
    except Exception as e:
        print(e)