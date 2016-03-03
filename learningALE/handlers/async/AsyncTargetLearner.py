from learningALE.handlers.async.AsyncClient import AsyncClient
from learningALE.handlers.actionhandler import ActionHandler
import numpy as np


class AsyncTargetLearner(AsyncClient):
    def __init__(self, num_actions, initial_cnn_values, cnn_partial, pipe,
                 skip_frame=4, phi_length=4, async_update_step=5, target_update_frames=40000):
        super().__init__(pipe)

        # initialize action handler, ending E-greedy is either 0.1, 0.01, 0.5 with probability 0.4, 0.3, 0.3
        end_rand = np.random.choice([0.1, 0.01, 0.5], p=[0.4, 0.3, 0.3])
        rand_vals = (1, end_rand, 4000000)  # anneal over four million frames
        self.action_handler = ActionHandler(rand_vals)

        # initialize cnn
        self.cnn = cnn_partial()
        self.cnn.set_parameters(initial_cnn_values)
        self.cnn.set_target_parameters(initial_cnn_values)
        self.frame_buffer = np.zeros((1, phi_length, 84, 84), dtype=np.float32)

        self.skip_frame = skip_frame
        self.phi_length = phi_length
        self.loss_list = list()

        self.async_update_step = async_update_step
        self.target_update_frames = target_update_frames
        self.target_update_count = 0

    def add_state_to_buffer(self, state):
        self.frame_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
        self.frame_buffer[0, self.phi_length-1] = state

    def frame_buffer_with(self, state):
        empty_buffer = np.zeros((1, self.phi_length, 84, 84), dtype=np.float32)
        empty_buffer[0, 0:self.phi_length-1] = self.frame_buffer[0, 1:self.phi_length]
        empty_buffer[0, self.phi_length-1] = state
        return empty_buffer

    def check_update_target(self, total_frames_count):
        if total_frames_count >= self.target_update_count * self.target_update_frames:
            self.target_update_count += 1
            return True
        return False

    def get_action(self, frame_buffer):
        return self.cnn.get_output(frame_buffer)[0]

    def get_game_action(self, frame_buffer):
        # checks to see if we are doing random, if so returns random game action
        rand, action = self.action_handler.get_random()
        if not rand:
            action = self.get_action(frame_buffer)
            return self.action_handler.action_vect_to_game_action(action, random=False)
        return action

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)
