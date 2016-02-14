import numpy as np
from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
from async_target_cnn import AsyncTargetCNN
from learningALE.handlers.async.PipeCmds import PipeCmds


class Async1StepDQNLearner:
    def __init__(self, num_actions, initial_cnn_values, pipe, random_state=np.random.RandomState()):
        super().__init__()

        # initialize action handler
        rand_vals = (1, 0.1, 100)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000 actions
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)

        self.cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions)
        self.cnn.set_parameters(initial_cnn_values)
        self.cnn.set_target_parameters(initial_cnn_values)

        self.frame_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)

        # client stuff
        self.thread_steps = 0
        self.pipe = pipe

    def add_state_to_buffer(self, state):
        self.frame_buffer[0, 0:2] = self.frame_buffer[0, 1:3]
        self.frame_buffer[0, 3] = state

    def frame_buffer_with(self, state):
        empty_buffer = np.zeros((1, 4, 84, 84))
        empty_buffer[0, 0:2] = self.frame_buffer[0, 1:3]
        empty_buffer[0, 3] = state
        return empty_buffer

    def game_over(self):
        self.frame_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)

    def run(self, emulator, skip_frame=4, async_update=5):
        # run until broken by pipe end command from host
        total_score = 0
        while True:
            # reset game
            print(self, 'starting episode. Step counter:', self.thread_steps, 'Last score:', total_score)
            emulator.reset()
            self.game_over()
            total_score = 0

            # get initial state
            state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # run until terminal
            terminal = False
            while not terminal:
                # get action
                action = self.get_game_action(state)

                # step and get new state
                reward = emulator.step(action, skip_frame=skip_frame, clip=1)
                total_score += reward
                state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

                # check for terminal
                terminal = emulator.get_game_over()

                # accumulate gradients
                self.cnn.accumulate_gradients(self.frame_buffer, self.action_handler.game_action_to_action_ind(action),
                                              reward, self.frame_buffer_with(state_tp1), terminal)

                state = state_tp1
                self.thread_steps += 1

                if self.thread_steps % async_update == 0 or terminal:
                    self.async_update()

                    # send my step count back to host
                    self.pipe.send((PipeCmds.ClientSendingSteps, self.thread_steps))

                    # if terminal check for host end
                    if terminal:
                        if self.pipe.poll():
                            pipe_cmd = self.pipe.recv()
                            if pipe_cmd == PipeCmds.End:
                                return
                            # if not end then we got something unexpected
                            else:
                                print('Dropping pipe command', pipe_cmd, 'and continuing')

    def async_update(self):
        self.pipe.send((PipeCmds.ClientSendingGradients, self.cnn.get_gradients()))

        # wait for host to send back new network parameters
        pipe_cmd, new_params = self.pipe.recv()

        # it's possible host will have sent updated target if so update target and re-wait for new parameters
        if pipe_cmd == PipeCmds.HostSendingGlobalTarget:
            self.cnn.set_target_parameters(new_params)
            # wait for host to send back new network parameters
            pipe_cmd, new_params = self.pipe.recv()

        if pipe_cmd == PipeCmds.HostSendingGlobalParameters:
            self.cnn.set_parameters(new_params)
            self.cnn.clear_gradients()

    def get_action(self):
        return self.cnn.get_output(self.frame_buffer)[0]

    def get_game_action(self, state):
        self.add_state_to_buffer(state)
        return self.action_handler.action_vect_to_game_action(
            self.get_action())

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)


from multiprocessing import Process
class Async1StepQLearnerProcess(Process):
    """
    pipe_conn : :class:`Connection`
        Pipe child connection to communicate with host
    learner_partial : partial(Learner, args*)
        Learner partial function to construct the learner
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
