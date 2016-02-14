from multiprocessing import Pipe
from learningALE.handlers.gamehandler import MinimalGameHandler
from onestep_dqn_client import Async1StepQLearnerProcess
from functools import partial
from learningALE.handlers.async.PipeCmds import PipeCmds


class Async1StepQLearnerHost:
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
    def __init__(self, host_cnn, learners, rom, show_rom):
        # create host cnn
        self.cnn = host_cnn
        # create partial function to create game handlers
        game_handler_partial = partial(MinimalGameHandler, rom, show_rom)
        # setup learners and emulators
        self.learner_pipes = list()
        self.learner_processes = list()
        for learner_partial in learners:
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_process = Async1StepQLearnerProcess(args=(child_conn, learner_partial, game_handler_partial))
            learner_process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(learner_process)

    def start(self):
        for learner in self.learner_pipes:
            learner.send(PipeCmds.Start)

        self.busy_wait()

    def busy_wait(self):
        while True:
            for learner in self.learner_pipes:
                if learner.poll():
                    self.process_pipe(learner)

    def process_pipe(self, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradients:
            self.cnn.gradient_step(extras)
            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters, self.cnn.get_parameters()))

    def block_until_done(self):
        self.end_processes()
        for learner in self.learner_processes:
            learner.join()

    def end_processes(self):
        for learner in self.learner_pipes:
            learner.send(PipeCmds.End)