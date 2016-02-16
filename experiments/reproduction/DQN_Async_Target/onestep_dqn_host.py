from multiprocessing import Pipe
from learningALE.handlers.gamehandler import MinimalGameHandler
from onestep_dqn_client import Async1StepQLearnerProcess
from functools import partial
from learningALE.handlers.async.PipeCmds import PipeCmds
import time
import pickle


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
    def __init__(self, host_cnn, learners, rom, show_rom, target_update=10000):
        # create host cnn
        self.cnn = host_cnn
        self.target_update = target_update
        # create partial function to create game handlers
        game_handler_partial = partial(MinimalGameHandler, rom, show_rom)
        # setup learners and emulators
        self.learner_pipes = list()
        self.learner_processes = list()
        self.learner_steps = list()
        self.learner_update_target_flag = list()
        self.learner_stats = list()
        for learner_partial in learners:
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_process = Async1StepQLearnerProcess(args=(child_conn, learner_partial, game_handler_partial))
            learner_process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(learner_process)
            self.learner_steps.append(0)
            self.learner_update_target_flag.append(False)
            self.learner_stats.append(list())

    def run(self, epochs=10):
        ep_count = 0
        for learner in self.learner_pipes:
            learner.send(PipeCmds.Start)

        st = time.time()
        while sum(self.learner_steps) < epochs * 50000:  # 50000 updates is defined as an epoch
            for learner_ind, learner in enumerate(self.learner_pipes):
                if learner.poll():
                    self.process_pipe(learner_ind, learner)
            if sum(self.learner_steps) >= ep_count * 25000 and len(self.learner_stats[-1]) > 0:
                self.print_status(st)
                with open('async1stepdqn{0}.pkl'.format(sum(self.learner_steps)), 'wb') as out_file:
                    pickle.dump(self.cnn.get_parameters(), out_file)
                ep_count += 0.5

    def process_pipe(self, learner_ind, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradientsSteps:
            self.cnn.gradient_step(extras[0])
            self.learner_steps[learner_ind] = extras[1]

            update_target = self.check_update_target(learner_ind)
            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters, (self.cnn.get_parameters(), update_target)))
        if pipe_cmd == PipeCmds.ClientSendingStats:
            self.learner_stats[learner_ind].append(extras)

    def print_status(self, st):
        frames = 0
        for learner_stat in self.learner_stats:
            if len(learner_stat) > 0:
                frames += learner_stat[-1]['frames']
        et = time.time()
        print('==== Status Report ====')
        print('Epoch:', round(float(sum(self.learner_steps)) / 50000, 1))
        print('Time:', et-st)
        print('Frames:', frames)
        print('FPS:', frames/(et-st))
        print('=======================')

    def check_update_target(self, learner_ind):
        if sum(self.learner_steps) % self.target_update == 0:
            for ind in range(len(self.learner_update_target_flag)):
                self.learner_update_target_flag[ind] = True
        update_target = self.learner_update_target_flag[learner_ind]
        self.learner_update_target_flag[learner_ind] = False
        return update_target

    def block_until_done(self):
        self.end_processes()
        for learner in self.learner_processes:
            learner.join()

    def end_processes(self):
        for learner in self.learner_pipes:
            learner.send((PipeCmds.End, None))
