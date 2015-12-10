__author__ = 'Ben'
from learningALE.handlers.binarysearch import BinaryTree
from learningALE.handlers.experiencehandler import ExperienceHandler
import numpy as np


class PrioritizedExperienceHandler(ExperienceHandler):
    def __init__(self, max_len, num_actions, discount, dtype=np.float32):
        super().__init__(max_len)
        self.num_actions = num_actions
        self.dtype = dtype
        self.discount = discount
        self.costList = list()
        self.tree = BinaryTree()
        self.buffer = 0

    def add_experience(self, state, action, reward):
        insert_ind = super().add_experience(state, action, reward)
        if insert_ind > 1:
            # insert into tree with np.inf so that it will guarantee to be looked at
            self.tree.insert(np.inf, insert_ind-2)  # -2 so we have the state tp1, and because insert_ind is len(list)

    def add_terminal(self):
        super().add_terminal()
        self.tree.insert(np.inf, self.num_inserted-1)

    def get_prioritized_experience(self, mini_batch, dtype=np.float32):
        if self.num_inserted <= mini_batch:
            return None, None, None, None, None, None

        state_shape = self.states[0].shape
        mb_states = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_states_tp1 = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_actions = list()
        mb_rewards = list()
        mb_terminal = list()
        mb_inds_poped = list()
        for exp in range(mini_batch):
            td_error, exp_ind = self.tree.pop_max()
            mb_inds_poped.append(exp_ind)
            mb_states[exp] = self.states[exp_ind]

            if exp_ind in self.term_states:
                mb_terminal.append(1)
            else:
                mb_terminal.append(0)
                mb_states_tp1[exp] = self.states[exp_ind+1]
            mb_actions.append(self.actions[exp_ind])
            mb_rewards.append(self.rewards[exp_ind])

        mb_actions = np.asarray(mb_actions, dtype=int)
        mb_rewards = np.asarray(mb_rewards, dtype=dtype)
        mb_terminal = np.asarray(mb_terminal, dtype=int)
        return mb_states, mb_actions, mb_rewards, mb_states_tp1, mb_terminal, mb_inds_poped

    def train(self, mini_batch, cnn):
        # generate minibatch data
        states, actions, rewards, state_tp1s, terminal, mb_inds_poped = self.get_prioritized_experience(mini_batch)

        if states is not None:
            r_tp1 = cnn.get_output(state_tp1s)
            max_tp1 = np.max(r_tp1, axis=1)
            rewards += (1-terminal) * self.discount * max_tp1

            rewVals = np.zeros((mini_batch, self.num_actions), dtype=self.dtype)
            arange = np.arange(mini_batch)
            rewVals[arange, actions] = rewards

            mask = np.zeros((mini_batch, self.num_actions), dtype=self.dtype)
            nonZero = np.where(rewVals != 0)
            mask[nonZero[0], nonZero[1]] = 1
            cost, output_states = cnn.train(states, rewVals, mask)
            self.costList.append(cost)

            # update tree with new td_error
            max_states = np.max(output_states, axis=1)
            td_errors = np.abs(max_tp1 - max_states)

            for td_error, ind in zip(td_errors, mb_inds_poped):
                self.tree.insert(td_error, ind)

    def trim(self):
        super().trim()