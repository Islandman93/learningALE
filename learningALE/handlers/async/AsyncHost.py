from functools import partial
from multiprocessing import Pipe
from learningALE.handlers.ale_specific.gamehandler import MinimalGameHandler
from learningALE.handlers.async.PipeCmds import PipeCmds
import json
import time
import pickle


class AsyncLearnerHost:
    """
    The :class:`AsyncLearnerHost` class is used to be able to run multiple learners on emulator instances, each
    learner will have it's own emulator instance. It should be used for learners that cannot be used outside the thread
    that spawns them (Theano). If your learner can be used outside of the spawning thread (everything else but Theano)
    use :class:`ThreadedGameHandler`.

    Parameters
    ----------
    host_cnn : NeuralNetwork
        The host acts as the shared parameters for the neural network used in all threads. It should be generally the
        same class of neural network used in the learner. All threads will initialize to its parameters.
    learners : Iterable<tuple(partial(learner_constructor, args), LearnerProcessClass)>
        The learner classes to create with their respective args
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Whether or not to show the game being played or not. True takes longer to run but can be fun to watch, the ALE
        display is not thread safe so it can be really fun to watch it jump between threads but crashes at some point
    skip_frame : int
        Number of frames to skip using the last action chosen
    """
    def __init__(self, host_cnn, learners, rom, skip_frame=4, show_rom=False, additional_process_args=None):
        # create host cnn
        self.cnn = host_cnn
        # create partial function to create game handlers
        game_handler_partial = partial(MinimalGameHandler, rom, skip_frame, show_rom)
        # setup learners and emulators
        self.learner_pipes = list()
        self.learner_processes = list()
        self.learner_frames = list()
        self.learner_stats = list()
        for learner_partial, process in learners:
            # create pipe
            parent_conn, child_conn = Pipe()

            # create and start child process to run constructors
            learner_process = process(args=(child_conn, learner_partial, game_handler_partial, additional_process_args), daemon=True)
            learner_process.start()

            self.learner_pipes.append(parent_conn)
            self.learner_processes.append(learner_process)
            self.learner_frames.append(0)
            self.learner_stats.append(list())

        self.best_score = 0

    def run(self, epochs=1, save_interval=1, show_status=True):
        ep_count = 0
        for learner in self.learner_pipes:
            learner.send(PipeCmds.Start)

        st = time.time()
        while sum(self.learner_frames) < epochs * 4000000:  # 4000000 frames is defined as an epoch
            for learner_ind, learner in enumerate(self.learner_pipes):
                if learner.poll():
                    self.process_pipe(learner_ind, learner)

            if sum(self.learner_frames) >= ep_count * 4000000:
                ep_count += save_interval

                # save network parms
                with open('async_network_parameters{0}.pkl'.format(sum(self.learner_frames)), 'wb') as out_file:
                    pickle.dump(self.cnn.get_parameters(), out_file)

                # save score and loss lists
                learner_stats = {}
                for learner_ind, learner_stat in enumerate(self.learner_stats):
                    if len(learner_stat) > 0:
                        scores = list()
                        loss = list()
                        for games in learner_stat:
                            scores.append(int(games['score']))  # score is np.int which is not serializable
                            loss += games['loss']
                        learner_stats[learner_ind] = {'scores': scores, 'loss': loss}
                with open('network_stats.json', 'w') as out_file:
                    json.dump(learner_stats, out_file)

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
        et = time.time()
        print('==== Status Report ====')
        print('Epoch:', round(float(sum(self.learner_frames)) / 4000000, 2))  # 4000000 frames is defined as an epoch
        print('Time:', et-st)
        print('Frames:', sum(self.learner_frames))
        print('FPS:', sum(self.learner_frames)/(et-st))
        print('Best score:', self.best_score)
        print('=======================')

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
