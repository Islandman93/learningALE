__author__ = 'Ben'
import numpy as np
import theano
import handlers.experiencehandler as exph
import copy


class RecurrentHandler(exph.ExperienceHandler):
    def __init__(self, miniBatch, discount, maxLen, numActions):
        super().__init__(miniBatch, discount, maxLen, numActions)
        self.bufStates = list()
        self.bufRewards = list()
        self.bufActions = list()

    def addToBuffer(self, state, action, reward):
        self.bufStates.append(state)
        self.bufActions.append(action)
        self.bufRewards.append(0)

        if reward != 0:
            self.processTD(reward)

    def processTD(self, reward):
        for revInd in range(len(self.bufRewards)-1, 0, -1):
            self.bufRewards[revInd] += reward
            reward *= self.discount

    def flushBuffer(self):
        self.states.append(copy.copy(self.bufStates))
        self.actions.append(copy.copy(self.bufActions))
        self.rewards.append(copy.copy(self.bufRewards))
        self.bufStates.clear()
        self.bufRewards.clear()
        self.bufActions.clear()

    def train_exp(self, cnn):
        cost = 0
        # generate minibatch data
        states, actions, rewards, _, isGood = self.getRandomExperience()

        if isGood:
            states = np.asarray(states)
            actions = np.asarray(actions, int)
            rewards = np.asarray(rewards, dtype=theano.config.floatX)

            # nonTerm = np.where(rewards == 0)
            # if nonTerm[0].size > 0:
            #     r_tp1 = cnn.get_output(state_tp1s[nonTerm[0]].reshape((-1,) + state_tp1s.shape[1:]))
            #     rewards[nonTerm[0]] = np.max(r_tp1, axis=1)*self.discount

            rewVals = np.zeros((states.shape[1], self.numActions), dtype=theano.config.floatX)
            arange = np.arange(states.shape[1])
            rewVals[arange, actions] = rewards

            mask = np.zeros((states.shape[1], self.numActions), dtype=theano.config.floatX)
            nonZero = np.where(rewVals != 0)
            mask[nonZero[0], nonZero[1]] = 1

            # reshape batch in
            states = states.reshape(1, states.shape[1], -1)
            rewVals = rewVals.reshape((1,) + rewVals.shape)
            mask = mask.reshape((1,) + mask.shape)
            cost += cnn.train(states, rewVals, mask)

        if cost > 0:
            self.costList.append(cost)

    def train_target(self, trainCount, cnn):
        pass

    def getRandomExperience(self):
        if len(self.states) < self.miniBatch:
            return None, None, None, None, 0

        rp = np.random.permutation(len(self.states))
        state = list()
        action = list()
        reward = list()
        for exp in range(self.miniBatch):
            state.append(self.states[rp[exp]])
            action.append(self.actions[rp[exp]])
            reward.append(self.rewards[rp[exp]])
        return state, action, reward, None, 1

    def getMultipleRandomExperience(self, numToGet):
        if len(self.states) < numToGet:
            return None, None, None, None, 0

        state = list()
        action = list()
        reward = list()
        state_tp1 = list()
        for exp in range(numToGet):
            states, actions, rewards, states_tp1, throwaway = self.getRandomExperience()
            state.append(states)
            action.append(actions)
            reward.append(rewards)
            state_tp1.append(states_tp1)
        return state, action, reward, state_tp1, 1

    def validation(self, cnn):
        # find values for rewards that are not zero
        rewards = np.asarray(self.rewards)
        non_zero_rew = np.where(rewards != 0)[0]
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
        one_hot = np.zeros((states.shape[0], self.numActions))
        one_hot[aran, actions] = rewards[non_zero_rew]
        loss = np.mean((output-one_hot)**2)

        return loss


import unittest


class TestExperienceHandler(unittest.TestCase):
    def test_train(self):
        expHandler = self.getExpHandler(101)
        class dummy:
            def train(self, x, y, mask):
                return np.random.randn(x.shape[0], 7)
        expHandler.train_exp(dummy())

    def getExpHandler(self, numDataToGet):
        expHandler = RecurrentHandler(1, 0.95, 1000, 7, 16)
        for i in range(numDataToGet):
            states = np.random.random((4, 105, 80))
            rewards = 0
            if i % 10 == 0 and i > 0:
                rewards = 1
            actions = np.random.randint(0, 7)
            expHandler.addToBuffer(states, actions, rewards)
        expHandler.processTD(-1)
        expHandler.flushBuffer()
        for i in range(numDataToGet):
            states = np.random.random((4, 105, 80))
            rewards = 0
            if i % 10 == 0 and i > 0:
                rewards = 1
            actions = np.random.randint(0, 7)
            expHandler.addToBuffer(states, actions, rewards)
        expHandler.processTD(-1)
        expHandler.flushBuffer()
        return expHandler

if __name__ == '__main__':
    unittest.main()