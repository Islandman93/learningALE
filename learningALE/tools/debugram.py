__author__ = 'Ben'


def main():
    import time
    import numpy as np
    from learningALE.libs.ale_python_interface import ALEInterface

    # this script is used to try and find what ram index stores the number of lives for a game

    ale = ALEInterface(True)

    ale.loadROM(b'D:\\_code\\beam_rider.bin')

    (screen_width, screen_height) = ale.getScreenDims()
    legal_actions = ale.getLegalActionSet()

    frameCount = 0
    ramlist = list()
    st = time.time()
    for episode in range(1):
        total_reward = 0.0
        while not ale.game_over():
            a = legal_actions[np.random.randint(legal_actions.size)]
            ram = ale.getRAM()
            ramlist.append(ram)
            reward = ale.act(a)
            total_reward += reward
            frameCount += 1
        print("Episode " + str(episode) + " ended with score: " + str(total_reward))
        ale.reset_game()
    et = time.time()
    print(et-st, frameCount/(et-st))

    import matplotlib.pyplot as plt
    ramarray = np.asarray(ramlist)
    w = np.where(ramarray > 3)[1]
    ramarray[:, w] = 0
    plt.plot(ramarray)

    notZ = np.where(ramarray != 0)[1]
    unqNZ = np.unique(notZ)
    print(unqNZ)

    # lastram = ramarray[0]
    # ind = np.ones(ramarray.shape[1], dtype=int)
    # for i in range(1,ramarray.shape[0]):
    #     currram = ramarray[i]
    #     ind = np.where(currram[ind] <= lastram[ind])
