import threading
from queue import Queue

from learningALE.handlers.ale_specific.gamehandler import GameHandler


class ThreadedGameHandler:
    """
    The :class:`ThreadedGameHandler` class is used to be able to run multiple learners on multiple emulator instances.
    It uses :class:'GameHandler' to communicate between the ALE and learner

    Parameters
    ----------
    rom : byte string
        Specifies the directory to load the rom from. Must be a byte string: b'dir_for_rom/rom.bin'
    show_rom : boolean
        Whether or not to show the game being played or not. True takes longer to run but can be fun to watch
    skip_frame : int
        Number of frames to skip using the last action chosen
    num_emulators : int
        Number of emulators/threads to setup and run on
    """
    def __init__(self, rom, show_rom, skip_frame, num_emulators):
        # setup list of gamehandlers and their locks
        self.emulators = list()
        for emu in range(num_emulators):
            self.emulators.append((GameHandler(rom, show_rom, skip_frame), threading.Lock()))

        # setup thread queue
        self.queue = Queue()

        # lock for unlocking/locking emulators
        self.emulator_lock = threading.Lock()
        self.current_emulator = 0
        self.num_emulators = num_emulators

    def async_run_emulator(self, learner, done_fn):
        # push to queue
        self.queue.put(self._get_next_emulator())
        t = threading.Thread(target=self._thread_run_emulator, args=(learner, done_fn))
        t.daemon = True
        t.start()

    def _thread_run_emulator(self, learner, done_fn):
        # get an emulator
        emulator, emulator_lock = self.queue.get()
        with emulator_lock:
            total_reward = emulator.run_one_game(learner)
        done_fn(total_reward)
        self.queue.task_done()

    def block_until_done(self):
        self.queue.join()

    def _get_next_emulator(self):
        with self.emulator_lock:
            emulator = self.emulators[self.current_emulator]
            self.current_emulator += 1
            self.current_emulator %= self.num_emulators
        return emulator

    def get_legal_actions(self):
        return self.emulators[0][0].get_legal_actions()