from learningALE.handlers.async.AsyncClient import AsyncClientProcess
from nstep_a3c_client import AsyncNStepA3CLearner
from AsyncA3CHost import AsyncA3CLearnerHost
from learningALE.learners.async_a3c_cnn import AsyncA3CCNN


def main():
    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    gamename = 'breakout'
    num_actions = 4
    discount = 0.95

    from functools import partial

    cnn = AsyncA3CCNN((None, 4, 84, 84), num_actions, 5)

    learner_count = 8
    learners = list()
    for learner in range(learner_count):
        learner_process = (partial(AsyncNStepA3CLearner, num_actions, cnn.get_parameters(), discount=discount), AsyncClientProcess)
        learners.append(learner_process)

    host = AsyncA3CLearnerHost(cnn, learners, rom)

    import time
    st = time.time()
    host.run(15, show_status=True)
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