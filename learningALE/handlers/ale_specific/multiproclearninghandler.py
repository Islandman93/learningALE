from functools import partial
from multiprocessing import Process, Pipe

from learningALE.handlers.ale_specific.gamehandler import GameHandler


class MultiprocLearningHandler:
    """
    The :class:`MultiprocLearningHandler` class is used to be able to run multiple learners on emulator instances, each
    learner will have it's own emulator instance. It should be used for learners that cannot be used outside the thread
    that spawns them (Theano). If your learner can be used outside of the spawning thread (everything else but Theano)
    use :class:`ThreadedGameHandler`.

    Parameters
    ----------
    learners : Iterable<partial(learner_constructor, args)>
        The learner classes to create with their respective args
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Whether or not to show the game being played or not. True takes longer to run but can be fun to watch, the ALE
        display is not thread safe so it can be really fun to watch it jump between threads but crashes at some point
    skip_frame : int
        Number of frames to skip using the last action chosen
    """
    def __init__(self, learners, rom, show_rom, skip_frame):
        # create partial function to create game handlers
        game_handler_partial = partial(GameHandler, rom, show_rom, skip_frame)
        # setup learners and emulators
        self.learner_pipes = list()
        self.learner_processes = list()
        for learner_partial in learners:
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_process = TrainingProcess(args=(child_conn, learner_partial, game_handler_partial))
            learner_process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(learner_process)

    def run_all(self):
        for learner in self.learner_pipes:
            learner.send(PipeCmds.RunEpisode)

    def block_until_done(self):
        self.end_processes()
        for learner in self.learner_processes:
            learner.join()

    def end_processes(self):
        for learner in self.learner_pipes:
            learner.send(PipeCmds.End)


class TrainingProcess(Process):
    def run(self):
        # access thread args from http://stackoverflow.com/questions/660961/overriding-python-threading-thread-run
        pipe, learner_partial, emulator_partial = self._args

        # create learner and emulator
        emulator = emulator_partial()
        learner = learner_partial()
        learner.set_legal_actions(emulator.get_legal_actions())

        # wait for pipe commands in a do-while loop
        while True:
            pipe_cmd = pipe.recv()

            # if we are done then break
            if pipe_cmd == PipeCmds.End:
                pipe.close()
                break
            # else if run episode
            elif pipe_cmd == PipeCmds.RunEpisode:
                total_reward = emulator.run_one_game(learner)
                print(self, total_reward)
                pipe.send(PipeCmds.EpisodeDone)
            # else invalid pipe command? how did this happen
            else:
                raise ValueError('Invalid pipe command {0}'.format(pipe_cmd))

from enum import Enum
class PipeCmds(Enum):
    End = 1
    RunEpisode = 2
    EpisodeDone = 3