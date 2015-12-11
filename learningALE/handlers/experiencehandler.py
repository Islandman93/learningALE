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
        if self.num_inserted < mini_batch:
            return None, None, None, None, None

        rp = np.random.choice(self.num_inserted, replace=False, size=mini_batch)
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
            # if we are deleting a state we need to decrement the terminal state indicators
            term_states = list(self.term_states)
            term_states = [x - 1 for x in term_states]
            self.term_states = term_states

            self.num_inserted -= 1

        while len(self.rewards) > self.max_len:
            del self.rewards[0]

        while len(self.actions) > self.max_len:
            del self.actions[0]