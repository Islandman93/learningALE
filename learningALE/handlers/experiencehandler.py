__author__ = 'Ben'
import numpy as np


class ExperienceHandler:
    """
    The :class:`ExperienceHandler` class handles storing of experience used for experience replay techniques. It keeps a
    list of states, rewards, actions, and terminal states. Design decisions: This class is relatively optimized, Lists
    were chosen because they are easy to append to and can be easily converted to numpy arrays for training.

    Parameters
    ----------
    max_len : int
       Specifies the maximum number of states/rewards/actions to store.
    """
    def __init__(self, max_len):
        self.states = list()
        self.rewards = list()
        self.actions = list()
        self.term_states = set()
        self.size = 0
        self.max_len = max_len

    def add_experience(self, state, action, reward):
        """
        Adds an experience (state, action, reward) to their respective lists

        Parameters
        ----------
        state : numpy.array
         Represents game state
        action : int
         Action_ind from learner
        reward : int
         Reward from ALE

        Returns
        -------
        size : int
            Current size of stored experiences
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.size += 1
        return self.size

    def add_terminal(self):
        """
        Adds a terminal state indicator at the last inserted state
        """
        self.term_states.add(len(self.states)-1)  # add a terminal state indicator.

    def get_random_experience(self, mini_batch, dtype=np.float32):
        """
        Used to get a random experience from memory.
        Returns numpy arrays representing state, action, reward, state time+1, terminal indicators, and indexes of where
        these are stored in this class' lists. Can return None if size < mini_batch requested. Otherwise returns random
        sampling (with replacement) of stored experiences in the format: (states, actions, rewards, states time+1,
        terminal, indexes).

        Parameters
        ----------
        mini_batch : int
         Number of experiences to get
        dtype : numpy data type
         Automatically converts returned states & states time+1 in this datatype

        Returns
        -------
        states : numpy.array(dtype)
         Numpy.array of dtype with shape (mini_batch,) + (state.shape)
        actions : numpy.array(int)
         Learner action_inds of the action performed
        rewards : numpy.array(int)
         Reward from ALE
        states_tp1 : numpy.array(dtype)
         Numpy.array of dtype with shape (mini_batch,) + (state.shape). Will be zeros if terminal is 1 for the
         corresponding state
        terminal : numpy.array(bool)
         Whether a state is terminal or not
        indexes : numpy.array(int)
         Index of where this return is stored in lists. Mostly unused except for testing
        """
        if self.size <= mini_batch:
            return None, None, None, None, None, None

        # Design decision: I know this can return duplicates, but is unlikely when size is large
        # It's also much more efficient see test/randint_vs_choice
        rp = np.random.randint(self.size - 1, size=mini_batch)
        state_shape = self.states[0].shape
        mb_states = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_states_tp1 = np.zeros((mini_batch,) + state_shape, dtype=dtype)
        mb_actions = list()
        mb_rewards = list()
        mb_terminal = list()
        mb_inds = list()
        for exp in range(mini_batch):
            exp_ind = rp[exp]
            mb_inds.append(exp_ind)
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
        mb_inds = np.asarray(mb_inds, dtype=int)
        return mb_states, mb_actions, mb_rewards, mb_states_tp1, mb_terminal, mb_inds

    def trim(self):
        """
        Trims/removes first inserted values until self.size < self.max_len. Moves terminal indicators to their new
        respective indexes. This code is not meant to be fast so it should not be called each timestep.
        """
        while len(self.states) > self.max_len:
            del self.states[0]
            # if we are deleting a state we need to decrement the terminal state indicators
            term_states = np.asarray(list(self.term_states))
            term_states -= 1
            term_states = term_states[term_states >= 0]
            self.term_states = set(term_states)

            self.size -= 1

        while len(self.rewards) > self.max_len:
            del self.rewards[0]

        while len(self.actions) > self.max_len:
            del self.actions[0]