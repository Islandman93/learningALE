from learningALE.handlers.async.AsyncHost import AsyncLearnerHost
from learningALE.handlers.async.AsyncClient import AsyncClientProcess
from onestep_dqn_client import Async1StepDQNLearner
from onestep_sarsa_client import Async1StepSarsaLearner
from learningALE.learners.async_target_cnn import AsyncTargetCNN, AsyncTargetCNNSarsa


def main():
    # setup vars
    rom = b'D:\\_code\\breakout.bin'
    gamename = 'breakout'
    num_actions = 4
    discount = 0.95
    sarsa = True

    from functools import partial

    if sarsa:
        cnn = AsyncTargetCNNSarsa((1, 4, 84, 84), num_actions, discount)
    else:
        cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions, discount)

    learner_count = 8
    learners = list()
    for learner in range(learner_count):
        if sarsa:
            learner_process = (partial(Async1StepSarsaLearner, num_actions, cnn.get_parameters()), AsyncClientProcess)
            learners.append(learner_process)
        else:
            learner_process = (partial(Async1StepDQNLearner, num_actions, cnn.get_parameters()), AsyncClientProcess)
            learners.append(learner_process)

    host = AsyncLearnerHost(cnn, learners, rom)

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