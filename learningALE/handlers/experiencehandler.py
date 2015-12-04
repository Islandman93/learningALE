__author__ = 'Ben'
import numpy as np


class ExperienceHandler:
    def __init__(self, max_len):
        self.states = list()
        self.rewards = list()
        self.actions = list()
        self.max_len = max_len

    def add_experience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if len(self.states) > self.max_len:
            del self.states[0]

        if len(self.rewards) > self.max_len:
            del self.rewards[0]

        if len(self.actions) > self.max_len:
            del self.actions[0]

    def get_random_experience(self, mini_batch, dtype=np.float32):
        if len(self.states) <= mini_batch:
            return None, None, None, None, 0

        rp = np.random.permutation(len(self.states)-1)
        state = list()
        action = list()
        reward = list()
        state_tp1 = list()
        for exp in range(mini_batch):
            state.append(self.states[rp[exp]])
            action.append(self.actions[rp[exp]])
            reward.append(self.rewards[rp[exp]])
            state_tp1.append(self.states[rp[exp]+1])

        state = np.asarray(state)
        action = np.asarray(action, int)
        reward = np.asarray(reward, dtype=dtype)
        state_tp1 = np.asarray(state_tp1, dtype=dtype)
        return state, action, reward, state_tp1, 1


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