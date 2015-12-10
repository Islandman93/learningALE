__author__ = 'Ben'
import numpy as np


class ExperienceHandler:
    def __init__(self, max_len):
        self.states = list()
        self.rewards = list()
        self.actions = list()
        self.term_states = set()
        self.num_inserted = 0
        self.max_len = max_len

    def add_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.num_inserted += 1
        return self.num_inserted

    def add_terminal(self):
        self.term_states.add(len(self.states)-1)  # add a terminal state indicator.

    # @profile
    def get_random_experience(self, mini_batch, dtype=np.float32):
        if self.num_inserted <= mini_batch:
            return None, None, None, None, None

        rp = np.random.permutation(len(self.states)-1)
        state_shape = self.states[0].shape
        mb_states = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_states_tp1 = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_actions = list()
        mb_rewards = list()
        mb_terminal = list()
        for exp in range(mini_batch):
            exp_ind = rp[exp]
            mb_states[exp] = self.states[exp_ind]
            mb_actions.append(self.actions[exp_ind])
            mb_rewards.append(self.rewards[exp_ind])
            if exp_ind in self.term_states:
                mb_terminal.append(1)
            else:
                mb_terminal.append(0)
                mb_states_tp1[exp] = self.states[exp_ind+1]

        mb_actions = np.asarray(mb_actions, dtype=int)
        mb_rewards = np.asarray(mb_rewards, dtype=dtype)
        mb_terminal = np.asarray(mb_terminal, dtype=int)
        return mb_states, mb_actions, mb_rewards, mb_states_tp1, mb_terminal

    def trim(self):
        while len(self.states) > self.max_len:
            del self.states[0]
            self.term_states -= 1  # if we are deleting a state we need to decrement the terminal state indicators

        while len(self.rewards) > self.max_len:
            del self.rewards[0]

        while len(self.actions) > self.max_len:
            del self.actions[0]


import unittest


class TestExperienceHandler(unittest.TestCase):
    def test_validation(self):
        expHandler = self.getExpHandler(100)
        class dummy:
            def get_output(self, x):
                return np.random.randn(x.shape[0], 7)

    def getExpHandler(self, numDataToGet):
        expHandler = ExperienceHandler(32, 0.95, 1000, 7)
        for i in range(numDataToGet):
            states = np.random.random((4, 105, 80))
            rewards = np.random.randint(0, 2)
            actions = np.random.randint(0, 7)
            expHandler.add_experience(states, actions, rewards)
        return expHandler

if __name__ == '__main__':
    unittest.main()