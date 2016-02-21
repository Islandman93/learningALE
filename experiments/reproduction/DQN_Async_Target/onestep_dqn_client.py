import numpy as np
from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
from async_target_cnn import AsyncTargetCNN
from learningALE.handlers.async.PipeCmds import PipeCmds


class Async1StepDQNLearner:
    def __init__(self, num_actions, initial_cnn_values, pipe, skip_frame=4, async_update_step=5, target_update_frames=40000,
                 random_state=np.random.RandomState()):
        super().__init__()

        # initialize action handler, ending E-greedy is either 0.1, 0.01, 0.5 with probability 0.4, 0.3, 0.3
        end_rand = np.random.choice([0.1, 0.01, 0.5], p=[0.4, 0.3, 0.3])
        rand_vals = (1, end_rand, 4000000/skip_frame)  # anneal over four million frames (frames/skip_frame)
        self.action_handler = ActionHandler(rand_vals)

        self.cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions)
        self.cnn.set_parameters(initial_cnn_values)
        self.cnn.set_target_parameters(initial_cnn_values)
        self.frame_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)

        self.skip_frame = skip_frame
        self.async_update_step = async_update_step
        self.target_update_step = target_update_frames/skip_frame  # the paper states target updates in frames not steps
        self.target_update_count = 0

        self.loss_list = list()

        # client stuff
        self.thread_steps = 0
        self.pipe = pipe
        self.done = False

    def run(self, emulator):
        # run until broken by pipe end command from host
        total_score = 0
        while not self.done:
            # reset game
            print(self, 'starting episode. Step counter:', self.thread_steps,
                  'Last score:', total_score, 'Current Rand Val:', self.action_handler.curr_rand_val)
            emulator.reset()
            self.game_over()
            total_score = 0

            # get initial state
            state = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # run until terminal
            terminal = False
            while not terminal and not self.done:
                # get action
                action = self.get_game_action(state)

                # step and get new state
                reward = emulator.step(action, skip_frame=self.skip_frame, clip=1)
                total_score += reward
                state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

                # check for terminal
                terminal = emulator.get_game_over()

                # accumulate gradients
                loss = self.cnn.accumulate_gradients(self.frame_buffer, self.action_handler.game_action_to_action_ind(action),
                                              reward, self.frame_buffer_with(state_tp1), terminal)
                self.loss_list.append(loss)

                state = state_tp1
                self.thread_steps += 1

                if self.thread_steps % self.async_update_step == 0 or terminal:
                    # process cmds from host, this will flush pipe recv commands
                    self.process_host_cmds()

                    # synchronous update parameters, this sends then waits for host to send back
                    self.synchronous_update()

                    # if terminal send stats
                    if terminal:
                        stats = {'score': total_score, 'frames': self.thread_steps*self.skip_frame, 'loss': self.loss_list}
                        self.pipe.send((PipeCmds.ClientSendingStats, stats))
        print(self, 'Stopping')

    def synchronous_update(self):
        # send accumulated grads
        self.pipe.send((PipeCmds.ClientSendingGradientsSteps, (self.cnn.get_gradients(), self.thread_steps*self.skip_frame)))
        self.cnn.clear_gradients()

        pipe_cmd, extras = self.pipe.recv()
        if pipe_cmd == PipeCmds.HostSendingGlobalParameters:
            (new_params, global_vars) = extras
            self.cnn.set_parameters(new_params)
            self.update_global_vars(global_vars)
            if self.check_update_target():
                print(self, 'setting target')
                self.cnn.set_target_parameters(new_params)

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

    def update_global_vars(self, global_vars):
        self.global_counter = global_vars['counter']
        self.action_handler.anneal_to(self.global_counter)

    def check_update_target(self):
        if self.global_counter >= self.target_update_count * self.target_update_step:
            self.target_update_count += 1
            return True
        return False

    def add_state_to_buffer(self, state):
        self.frame_buffer[0, 0:3] = self.frame_buffer[0, 1:4]
        self.frame_buffer[0, 3] = state

    def frame_buffer_with(self, state):
        empty_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)
        empty_buffer[0, 0:3] = self.frame_buffer[0, 1:4]
        empty_buffer[0, 3] = state
        return empty_buffer

    def game_over(self):
        self.frame_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)
        self.loss_list = list()

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
        Learner partial function to construct the learner. Pipe to host will be passed in as last var
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


class Async1StepDQNTester(Async1StepDQNLearner):
    def __init__(self, load_file, num_actions, skip_frame, rand_val=0.05):
        rand_vals = (rand_val, rand_val, 2)
        self.action_handler = ActionHandler(rand_vals)

        self.cnn = AsyncTargetCNN((1, 4, 84, 84), num_actions)
        import pickle
        with open(load_file, 'rb') as inp_file:
            self.cnn.set_parameters(pickle.load(inp_file))

        self.frame_buffer = np.zeros((1, 4, 84, 84), dtype=np.float32)
        self.skip_frame = skip_frame

    def run(self, emulator):
        # reset game
        print(self, 'starting episode. Current Rand Val:', self.action_handler.curr_rand_val)
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
            reward = emulator.step(action, skip_frame=self.skip_frame, clip=1)
            total_score += reward
            state_tp1 = np.asarray(emulator.get_gamescreen()/255.0, dtype=np.float32)

            # check for terminal
            terminal = emulator.get_game_over()

            # accumulate gradients
            loss = self.cnn.accumulate_gradients(self.frame_buffer, self.action_handler.game_action_to_action_ind(action),
                                          reward, self.frame_buffer_with(state_tp1), terminal)
            self.loss_list.append(loss)

            state = state_tp1
        return total_score