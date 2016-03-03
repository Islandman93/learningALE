import numpy as np
from learningALE.handlers.actionhandler import ActionHandler
from learningALE.handlers.async.PipeCmds import PipeCmds


class AsyncClient:
    def __init__(self, pipe):
        # client stuff
        self.thread_steps = 0
        self.pipe = pipe
        self.done = False

    def run(self, emulator):
        while not self.done:
            self.run_episode(emulator)

    def run_episode(self, emulator):
        raise NotImplementedError("subclasses must implement the run function run_episode(emulator)")

    def synchronous_update(self, gradients, frames, extra_parms=None):
        # send accumulated grads
        self.pipe.send((PipeCmds.ClientSendingGradientsSteps, (gradients, frames, extra_parms)))

        pipe_cmd, extras = self.pipe.recv()
        if pipe_cmd == PipeCmds.HostSendingGlobalParameters:
            (new_params, global_vars) = extras
            return new_params, global_vars

    def process_host_cmds(self):
        pipe_recieved = False
        while self.pipe.poll():
            pipe_cmd, extras = self.pipe.recv()
            pipe_recieved = True
            if pipe_cmd == PipeCmds.End:
                self.done = True
            else:
                print('Dropping pipe command', pipe_cmd, 'and continuing')
        return pipe_recieved


from multiprocessing import Process
class AsyncClientProcess(Process):
    """
    pipe_conn : :class:`Connection`
        Pipe child connection to communicate with host
    learner_partial : partial(Learner, args*)
        Learner partial function to construct the learner. Pipe to host will be passed in as last var
    emulator_partial : partial(Emulator, args*)
        Emulator partial function to construct the emulator. No additional args will be passed
    """
    def run(self):
        # access thread args from http://stackoverflow.com/questions/660961/overriding-python-threading-thread-run
        pipe, learner_partial, emulator_partial = self._args

        # create learner and emulator
        emulator = emulator_partial()
        learner = learner_partial(pipe)
        learner.set_legal_actions(emulator.get_legal_actions())

        # wait for start command
        pipe.recv()
        learner.run(emulator)