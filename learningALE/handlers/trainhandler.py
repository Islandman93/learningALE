import numpy as np


class TrainHandler:
    def __init__(self, mini_batch, discount, num_actions, dtype=np.float32):
        self.costList = list()
        self.mini_batch = mini_batch
        self.discount = discount
        self.num_actions = num_actions
        self.dtype = dtype

    def train_exp(self, exp_handler, cnn):
        # generate minibatch data
        states, actions, rewards, state_tp1s, isGood = exp_handler.get_random_experience(self.mini_batch)

        if isGood:
            nonTerm = np.where(rewards == 0)
            r_tp1 = cnn.get_output(state_tp1s[nonTerm[0]])
            rewards[nonTerm[0]] = np.max(r_tp1, axis=1)*self.discount

            self._train_from_ras(rewards, actions, states, cnn)

    def train_target(self, exp_handler, cnn, target):
        # generate minibatch data
        states, actions, rewards, state_tp1s, isGood = exp_handler.get_random_experience(self.mini_batch)

        if isGood:
            nonTerm = np.where(rewards == 0)
            r_tp1 = target.get_output(state_tp1s[nonTerm[0]])
            rewards[nonTerm[0]] = np.max(r_tp1, axis=1)*self.discount

            self._train_from_ras(rewards, actions, states, cnn)

    def train_double(self, exp_handler, cnn, target):
        # generate minibatch data
        states, actions, rewards, state_tp1s, isGood = exp_handler.get_random_experience(self.mini_batch)

        if isGood:
            nonTerm = np.where(rewards == 0)
            action_selection = np.argmax(cnn.get_output(state_tp1s[nonTerm[0]]), axis=1)
            value_estimation = target.get_output(state_tp1s[nonTerm[0]])
            r_double_dqn = value_estimation[np.arange(0, action_selection.size), action_selection]
            rewards[nonTerm[0]] = r_double_dqn*self.discount

            self._train_from_ras(rewards, actions, states, cnn)

    def _train_from_ras(self, rewards, actions, states, cnn):
        rewVals = np.zeros((self.mini_batch, self.num_actions), dtype=self.dtype)
        arange = np.arange(self.mini_batch)
        rewVals[arange, actions] = rewards

        mask = np.zeros((self.mini_batch, self.num_actions), dtype=self.dtype)
        nonZero = np.where(rewVals != 0)
        mask[nonZero[0], nonZero[1]] = 1
        cost = cnn.train(states, rewVals, mask)

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