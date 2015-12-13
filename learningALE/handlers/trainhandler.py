import numpy as np
from learningALE.handlers.experiencehandler import ExperienceHandler
from learningALE.handlers.prioritizedexperiencehandler import PrioritizedExperienceHandler


class TrainHandler:
    def __init__(self, mini_batch, discount, num_actions, dtype=np.float32):
        self.costList = list()
        self.mini_batch = mini_batch
        self.discount = discount
        self.num_actions = num_actions
        self.dtype = dtype

    def train_exp(self, exp_handler: ExperienceHandler, cnn):
        # generate minibatch data
        states, actions, rewards, state_tp1s, terminal = exp_handler.get_random_experience(self.mini_batch)

        if states is not None:
            r_tp1 = cnn.get_output(state_tp1s)
            rewards += (1-terminal) * self.discount * np.max(r_tp1, axis=1)

            self._train_from_ras(rewards, actions, states, cnn)

    def train_target(self, exp_handler: ExperienceHandler, cnn, target):
        # generate minibatch data
        states, actions, rewards, state_tp1s, terminal = exp_handler.get_random_experience(self.mini_batch)

        if states is not None:
            r_tp1 = target.get_output(state_tp1s)
            rewards += (1-terminal) * self.discount * np.max(r_tp1, axis=1)

            self._train_from_ras(rewards, actions, states, cnn)

    def train_double(self, exp_handler: ExperienceHandler, cnn, target):
        # generate minibatch data
        states, actions, rewards, state_tp1s, terminal = exp_handler.get_random_experience(self.mini_batch)

        if states is not None:
            action_selection = np.argmax(cnn.get_output(state_tp1s), axis=1)
            value_estimation = target.get_output(state_tp1s)
            r_double_dqn = value_estimation[np.arange(0, self.mini_batch), action_selection]
            rewards += (1-terminal) * self.discount * r_double_dqn

            self._train_from_ras(rewards, actions, states, cnn)

    def train_prioritized_exp(self, exp_handler: PrioritizedExperienceHandler, mini_batch: int, cnn):
        # generate minibatch data
        states, actions, rewards, state_tp1s, terminal, mb_inds_popped = exp_handler.get_prioritized_experience(mini_batch)

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

            # update prioritized exp handler with new td_error
            max_states = np.max(output_states, axis=1)
            td_errors = np.abs(max_tp1 - max_states)

            exp_handler.set_new_td_errors(td_errors, mb_inds_popped)

    def _train_from_ras(self, rewards, actions, states, cnn):
        rewVals = np.zeros((self.mini_batch, self.num_actions), dtype=self.dtype)
        arange = np.arange(self.mini_batch)
        rewVals[arange, actions] = rewards

        mask = np.zeros((self.mini_batch, self.num_actions), dtype=self.dtype)
        nonZero = np.where(rewVals != 0)
        mask[nonZero[0], nonZero[1]] = 1
        cost, states_output = cnn.train(states, rewVals, mask)

        self.costList.append(cost)

    def validation(self, cnn):
        # find values for rewards that are not zero
        rewards = np.asarray(self.rewards)
        non_zero_rew = np.where(rewards > 0)[0]
        if len(non_zero_rew) > 1000:
            rp = np.random.permutation(non_zero_rew)
            non_zero_rew = rp[0:1000]

        # get states corresponding to non zero rewards
        states = np.asarray(self.states)[non_zero_rew]

        # get actions
        actions = np.asarray(self.actions)[non_zero_rew]

        # get output from net
        output = cnn.get_output(states)

        # convert actions to one hot
        aran = np.arange(states.shape[0])
        one_hot = np.zeros((states.shape[0], self.num_actions))
        one_hot[aran, actions] = rewards[non_zero_rew]
        mask = np.zeros(one_hot.shape)
        nonZ = np.where(one_hot != 0)
        mask[nonZ[0], nonZ[1]] = 1
        loss = np.mean(mask*(output-one_hot)**2)

        return loss