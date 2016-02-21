from multiprocessing import Pipe
from learningALE.handlers.gamehandler import MinimalGameHandler
from onestep_dqn_client import Async1StepQLearnerProcess
from functools import partial
from learningALE.handlers.async.PipeCmds import PipeCmds
import numpy as np
import matplotlib.pyplot as plt
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
        self.learner_frames = list()
        self.learner_stats = list()
        for learner_partial in learners:
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_process = Async1StepQLearnerProcess(args=(child_conn, learner_partial, game_handler_partial), daemon=True)
            learner_process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(learner_process)
            self.learner_frames.append(0)
            self.learner_stats.append(list())

        self.best_score = 0

    def run(self, epochs=1, show_status=True):
        ep_count = 0
        for learner in self.learner_pipes:
            learner.send(PipeCmds.Start)

        st = time.time()
        while sum(self.learner_frames) < epochs * 4000000:  # 4000000 frames is defined as an epoch
            for learner_ind, learner in enumerate(self.learner_pipes):
                if learner.poll():
                    self.process_pipe(learner_ind, learner)

            if sum(self.learner_frames) >= ep_count * 4000000:
                with open('async1stepdqn{0}.pkl'.format(sum(self.learner_frames)), 'wb') as out_file:
                    pickle.dump(self.cnn.get_parameters(), out_file)
                ep_count += 0.1
                if show_status:
                    self.print_status(st)

    def process_pipe(self, learner_ind, pipe):
        pipe_cmd, extras = pipe.recv()
        if pipe_cmd == PipeCmds.ClientSendingGradientsSteps:
            self.cnn.gradient_step(extras[0])
            self.learner_frames[learner_ind] = extras[1]
            # send back new parameters to client
            pipe.send((PipeCmds.HostSendingGlobalParameters,
                       (self.cnn.get_parameters(), {'counter': sum(self.learner_frames)})))
        if pipe_cmd == PipeCmds.ClientSendingStats:
            self.learner_stats[learner_ind].append(extras)
            if extras['score'] > self.best_score:
                self.best_score = extras['score']

    def print_status(self, st):
        plt.clf()
        for learner_ind, learner_stat in enumerate(self.learner_stats):
            if len(learner_stat) > 0:
                scores = list()
                loss = list()
                for learner in learner_stat:
                    scores.append(learner['score'])
                    loss += learner['loss']

                plt.subplot(len(self.learner_processes), 2, (learner_ind * 2) + 1)
                plt.plot(scores, '.')
                plt.ylim([0, max(scores)])
                plt.subplot(len(self.learner_processes), 2, (learner_ind * 2) + 2)
                plt.plot(loss, '.')
                plt.ylim([0, np.percentile(loss, 90)])
        et = time.time()
        print('==== Status Report ====')
        print('Epoch:', round(float(sum(self.learner_frames)) / 4000000, 1))  # 4000000 frames is defined as an epoch
        print('Time:', et-st)
        print('Frames:', sum(self.learner_frames))
        print('FPS:', sum(self.learner_frames)/(et-st))
        print('Best score:', self.best_score)
        print('=======================')
        plt.ion()
        plt.show()
        plt.pause(0.01)
        plt.ioff()

    def block_until_done(self):
        self.end_processes()
        for learner in self.learner_processes:
            if not learner.join(5):
                print("Can't join learner", learner)

    def end_processes(self):
        for learner in self.learner_pipes:
            # send command twice just in case
            learner.send((PipeCmds.End, None))
            learner.send((PipeCmds.End, None))
