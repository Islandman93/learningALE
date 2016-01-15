import numpy as np
from scipy.misc import imresize
from learningALE.libs.ale_python_interface import ALEInterface
__author__ = 'Ben'


class GameHandler:
    """
    The :class:`GameHandler` class takes care of the interface between the ALE and the learner.

    Currently supported is the ability to display the ALE screen when running, skip x number of frames by repeating the
    last action, and configuring the dtype to convert the gamescreen to (default is float16 for space).

    Parameters
    ----------
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Whether or not to show the game being played or not. True takes longer to run but can be fun to watch
    skip_frame : int
        Number of frames to skip using the last action chosen
    learner : :class:`learners.learner`
        Default None. The learner, on construction GameHandler will call set_legal_actions. If none then set_legal_actions
        needs to be called
    dtype : data type
        Specifies the data type to convert the game screen to. Default is np.float16
    """
    def __init__(self, rom, show_rom, skip_frame, learner=None, dtype=np.float16):
        # set up emulator
        self.ale = ALEInterface(show_rom)
        self.ale.loadROM(rom)
        (self.screen_width, self.screen_height) = self.ale.getScreenDims()
        legal_actions = self.ale.getMinimalActionSet()

        # set up vars
        self.skipFrame = skip_frame

        if learner:
            learner.set_legal_actions(legal_actions)

        self.frameCount = 0
        self.dtype = dtype

    def run_one_game(self, learner, neg_reward=True, early_return=False, clip=True):
        """
        Runs a full game. If lives and life_ram_ind are set then will give negative rewards on loss of life

        Parameters
        ----------
        learner : :class:`learners.learner`
            Will call get_game_action and frames_processed.
            get_game_action must return a valid ALE action ind. frames_processed can be a pass.

        neg_reward : bool
            Whether or not to use negative rewards, recieved when agent looses a life

        early_return : bool
            If set to true and lives/life_ram_ind are set then will return on first loss of life
        clip : bool
            Whether or not to clip positive rewards to 1

        Returns
        -------
        int
            Total reward from game. Can be negative if lives/life_ram_ind is set.
        """
        total_reward = 0.0
        gamescreen = None
        self.ale.reset_game()
        cur_lives = self.ale.lives()
        action_to_perform = 0  # initially set at zero because we start the game before asking the learner
        while not self.ale.game_over():
            # get frames
            frames = list()
            reward = 0

            # loop over skip frame
            for frame in range(self.skipFrame):
                gamescreen = self.ale.getScreenGrayscale(gamescreen)

                # convert ALE gamescreen into usable image, scaled between 0 and 1
                processedImg = imresize(gamescreen[25:-12, :, 0], 0.5, interp='nearest')
                frames.append(processedImg)

                # act on the action to perform, should be ALE compatible action ind
                rew = self.ale.act(action_to_perform)

                # clip positive rewards to 1
                if rew > 0 and clip:
                    reward += 1
                else:
                    reward += rew

                # if allowing negative rewards, get RAM and see if lives has decreased
                if neg_reward:
                    new_lives = self.ale.lives()
                    if new_lives < cur_lives:
                        reward -= 1  # losing a life is a negative 1 reward
                        cur_lives = new_lives

            # end frame skip loop

            total_reward += reward
            frames = np.asarray(frames)

            # frames_processed must be here before action_to_perform gets overwritten.
            learner.frames_processed(frames, action_to_perform, reward)

            action_to_perform = learner.get_game_action(frames)

            self.frameCount += 1*self.skipFrame

            # if doing early return, end game on first loss of life
            if reward < 0 and early_return:
                return total_reward

        # end of game
        return total_reward

    def set_legal_actions(self, learner):
        learner.set_legal_actions(self.ale.getMinimalActionSet())


class GameHandlerRam(GameHandler):
    def __init__(self, rom, showRom, randVals, expHandler, skip_frame, dtype=np.float16):
        super().__init__(rom, showRom, randVals, expHandler, skip_frame, dtype)

    def run_one_game(self, addFrameFn, outputFn, lives, life_ram_ind, train=True, trainFn=None, early_return=False, negReward=False):
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
                    if ram[life_ram_ind] < lives:
                        reward -= 1
                        lives = ram[life_ram_ind] #73 is space invaders
                
                ram /= 255
                frames.append(ram)

            total_reward += reward
            frames = np.asarray(frames)

            addFrameFn(frames, actionInd, reward)

            actionVect = outputFn(frames.reshape((1, self.skipFrame, 128)))[0]
            self.actionHandler.setAction(actionVect)

            self.frameCount += 1*self.skipFrame
            if reward < 0 and early_return:
                return total_reward


        # end of game
        self.actionHandler.anneal()
        return total_reward

    def evaluate(self, output_fn, early_return=False):
        return self.run_one_game(output_fn, train=False, early_return=early_return)


class GameHandlerReccurentRam(GameHandler):
    def __init__(self, rom, showRom, randVals, expHandler, skip_frame, dtype=np.float16):
        super().__init__(rom, showRom, randVals, expHandler, skip_frame, dtype)

    def run_one_game(self, addFrameFn, outputFn, lives, life_ram_ind, train=True, trainFn=None, early_return=False, negReward=False):
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
                if ram[life_ram_ind] < lives:
                    reward -= 1
                    lives = ram[life_ram_ind] #73 is space invaders


            total_reward += reward
            frames = np.asarray(ram, dtype=self.dtype)/255

            addFrameFn(frames, actionInd, reward)
            gameStates.append(frames)

            netInp = np.asarray(gameStates).reshape((-1, len(gameStates), 128))

            actionVect = outputFn(netInp)[0]
            self.actionHandler.setAction(actionVect[-1])

            self.frameCount += 1*self.skipFrame
            if reward < 0 and early_return:
                return total_reward


        # end of game
        self.actionHandler.anneal()
        return total_reward

    def evaluate(self, output_fn, early_return=False):
        return self.run_one_game(output_fn, train=False, early_return=early_return)