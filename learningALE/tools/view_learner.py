__author__ = 'Ben'


def main():
    import os
    import pickle
    import time

    import lasagne
    import matplotlib.pyplot as plt
    import numpy as np
    from learners.nns import AlloEggoCnn
    from scipy.misc import imresize

    from learningALE.handlers import ActionHandler, ActionPolicy
    from learningALE.libs.ale_python_interface import ALEInterface

    dtype = np.float16
    plt.ion()

    # set up emulator
    ale = ALEInterface(True)
    ale.loadROM(b'D:\\_code\\breakout.bin')
    (screen_width, screen_height) = ale.getScreenDims()
    legal_actions = ale.getMinimalActionSet()
    lives = 5

    # set up vars
    skipFrame = 4

    actionHandler = ActionHandler(ActionPolicy.eGreedy, (0.1, 0.1, 2), legal_actions)
    scoreList = list()

    cnn = AlloEggoCnn()
    with open(os.getcwd() + '\saves\cnnbestalloego.pkl', 'rb') as fin:
        parms = pickle.load(fin)

    lasagne.layers.set_all_param_values(cnn.a_out, parms)

    frameCount = 0
    st = time.time()
    for episode in range(100):
        total_reward = 0.0
        while not ale.game_over():
            # get frames
            frames = list()
            reward = 0
            for frame in range(skipFrame):
                gamescreen = ale.getScreenRGB()
                processedImg = np.asarray(
                    imresize(gamescreen.view(np.uint8).reshape(screen_height, screen_width, 4)[:, :, 0], 0.5, interp='nearest'),
                    dtype=dtype)/255

                frames.append(processedImg)

                performedAction, actionInd = actionHandler.getLastAction()
                rew = ale.act(performedAction)
                if rew > 0:
                    reward += 1

                ram = ale.getRAM()
                if ram[57] != lives:
                    reward -= 1
                    lives = ram[57]

            frames = np.asarray(frames)

            actionVect = cnn.get_output(frames.reshape((1, skipFrame, 105, 80)))[0]
            actionHandler.setAction(actionVect)

            total_reward += reward
            frameCount += 1*skipFrame

        ale.reset_game()
        actionHandler.anneal()
        scoreList.append(total_reward)

        lives = 5



        print("Episode " + str(episode) + " ended with score: " + str(total_reward))

        et = time.time()
        print('Total Time:', et-st, 'Frame Count:', frameCount, 'FPS:',frameCount/(et-st))

    plt.clf()
    plt.plot(scoreList, '.')
    plt.pause(0.01)
    plt.ioff()
