__author__ = 'Ben'
import numpy as np
import time
from handlers.actionhandler import ActionHandler, ActionPolicy
from scipy.misc import imresize
from libs.ale_python_interface import ALEInterface


class BreakoutHandler:
    def __init__(self, rom, showRom, randVals, expHandler, skipFrame, dtype=np.float16):
        # set up emulator
        self.ale = ALEInterface(showRom)
        self.ale.loadROM(rom)
        (self.screen_width, self.screen_height) = self.ale.getScreenDims()
        legal_actions = self.ale.getMinimalActionSet()

        # set up vars
        self.skipFrame = skipFrame

        self.actionHandler = ActionHandler(ActionPolicy.eGreedy, randVals, legal_actions)
        self.expHandler = expHandler
        self.frameCount = 0
        self.dtype = dtype

    def runOneGame(self, addFrameFn, outputFn, lives, negRewardRamInd, train=True, trainFn=None, earlyReturn=False, negReward=False):
        total_reward = 0.0
        self.ale.reset_game()
        while not self.ale.game_over():
            if train:
                trainFn()

            # get frames
            frames = list()
            reward = 0
            for frame in range(self.skipFrame):
                gamescreen = self.ale.getScreenRGB()
                processedImg = np.asarray(
                    imresize(gamescreen.view(np.uint8).reshape(self.screen_height, self.screen_width, 4)[25:-12, :, 0], 0.5, interp='nearest'),
                    dtype=self.dtype)/255
                frames.append(processedImg)

                performedAction, actionInd = self.actionHandler.getLastAction()
                rew = self.ale.act(performedAction)
                if rew > 0:
                    reward += 1

                if negReward:
                    ram = self.ale.getRAM()
                    if ram[negRewardRamInd] < lives:
                        reward -= 1
                        lives = ram[negRewardRamInd] #73 is space invaders

            total_reward += reward
            frames = np.asarray(frames)

            addFrameFn(frames, actionInd, reward)

            actionVect = outputFn(frames.reshape((1, self.skipFrame, frames.shape[1], 80)))[0]
            self.actionHandler.setAction(actionVect)


            self.frameCount += 1*self.skipFrame
            if reward < 0 and earlyReturn:
                return total_reward


        # end of game
        self.actionHandler.gameOver()
        return total_reward

    def evaluate(self, outputFn, earlyReturn=False):
        return self.runOneGame(outputFn, train=False, earlyReturn=earlyReturn)


class BreakoutHandlerRam(BreakoutHandler):
    def __init__(self, rom, showRom, randVals, expHandler, skipFrame, dtype=np.float16):
        super().__init__(rom, showRom, randVals, expHandler, skipFrame, dtype)

    def runOneGame(self, addFrameFn, outputFn, lives, negRewardRamInd, train=True, trainFn=None, earlyReturn=False, negReward=False):
        total_reward = 0.0
        self.ale.reset_game()
        while not self.ale.game_over():
            if train:
                trainFn()

            # get ram frames
            frames = list()
            reward = 0
            for frame in range(self.skipFrame):
                ram = self.ale.getRAM()

                performedAction, actionInd = self.actionHandler.getLastAction()
                rew = self.ale.act(performedAction)
                if rew > 0:
                    reward += 1

                if negReward:                    
                    if ram[negRewardRamInd] < lives:
                        reward -= 1
                        lives = ram[negRewardRamInd] #73 is space invaders
                
                ram /= 255
                frames.append(ram)

            total_reward += reward
            frames = np.asarray(frames)

            addFrameFn(frames, actionInd, reward)

            actionVect = outputFn(frames.reshape((1, self.skipFrame, 128)))[0]
            self.actionHandler.setAction(actionVect)

            self.frameCount += 1*self.skipFrame
            if reward < 0 and earlyReturn:
                return total_reward


        # end of game
        self.actionHandler.gameOver()
        return total_reward

    def evaluate(self, outputFn, earlyReturn=False):
        return self.runOneGame(outputFn, train=False, earlyReturn=earlyReturn)


class BreakoutHandlerReccurentRam(BreakoutHandler):
    def __init__(self, rom, showRom, randVals, expHandler, skipFrame, dtype=np.float16):
        super().__init__(rom, showRom, randVals, expHandler, skipFrame, dtype)

    def runOneGame(self, addFrameFn, outputFn, lives, negRewardRamInd, train=True, trainFn=None, earlyReturn=False, negReward=False):
        total_reward = 0.0
        self.ale.reset_game()
        gameStates = list()
        while not self.ale.game_over():
            if train:
                trainFn()

            # get ram frames
            reward = 0
            ram = self.ale.getRAM()

            performedAction, actionInd = self.actionHandler.getLastAction()
            rew = self.ale.act(performedAction)
            if rew > 0:
                reward += 1

            if negReward:
                if ram[negRewardRamInd] < lives:
                    reward -= 1
                    lives = ram[negRewardRamInd] #73 is space invaders


            total_reward += reward
            frames = np.asarray(ram, dtype=self.dtype)/255

            addFrameFn(frames, actionInd, reward)
            gameStates.append(frames)

            netInp = np.asarray(gameStates).reshape((-1, len(gameStates), 128))

            actionVect = outputFn(netInp)[0]
            self.actionHandler.setAction(actionVect[-1])

            self.frameCount += 1*self.skipFrame
            if reward < 0 and earlyReturn:
                return total_reward


        # end of game
        self.actionHandler.gameOver()
        return total_reward

    def evaluate(self, outputFn, earlyReturn=False):
        return self.runOneGame(outputFn, train=False, earlyReturn=earlyReturn)