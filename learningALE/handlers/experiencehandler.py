__author__ = 'Ben'
import numpy as np
import theano


class ExperienceHandler:
    def __init__(self, miniBatch, discount, maxLen, numActions):
        self.states = list()
        self.rewards = list()
        self.actions = list()
        self.costList = list()
        self.maxLen = maxLen
        self.miniBatch = miniBatch
        self.discount = discount
        self.numActions = numActions

    def addExperience(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        if len(self.states) > self.maxLen:
            del self.states[0]

        if len(self.rewards) > self.maxLen:
            del self.rewards[0]

        if len(self.actions) > self.maxLen:
            del self.actions[0]

    def getRandomExperience(self):
        if len(self.states) <= self.miniBatch:
            return None, None, None, None, 0

        rp = np.random.permutation(len(self.states)-1)
        state = list()
        action = list()
        reward = list()
        state_tp1 = list()
        for exp in range(self.miniBatch):
            state.append(self.states[rp[exp]])
            action.append(self.actions[rp[exp]])
            reward.append(self.rewards[rp[exp]])
            state_tp1.append(self.states[rp[exp]+1])
        return state, action, reward, state_tp1, 1

    def train_exp(self, cnn):
        cost = 0
        # generate minibatch data
        states, actions, rewards, state_tp1s, isGood = self.getRandomExperience()

        if isGood:
            states = np.asarray(states)
            actions = np.asarray(actions, int)
            rewards = np.asarray(rewards, dtype=theano.config.floatX)
            state_tp1s = np.asarray(state_tp1s, dtype=theano.config.floatX)

            nonTerm = np.where(rewards == 0)
            r_tp1 = cnn.get_output(state_tp1s[nonTerm[0]])
            rewards[nonTerm[0]] = np.max(r_tp1, axis=1)*self.discount

            rewVals = np.zeros((self.miniBatch, self.numActions), dtype=theano.config.floatX)
            arange = np.arange(self.miniBatch)
            rewVals[arange, actions] = rewards

            mask = np.zeros((self.miniBatch, self.numActions), dtype=theano.config.floatX)
            nonZero = np.where(rewVals != 0)
            mask[nonZero[0], nonZero[1]] = 1
            cost += cnn.train(states, rewVals, mask)

        if cost > 0:
            self.costList.append(cost)

    def train_target(self, trainCount, cnn):
        cost = 0
        # generate batch data
        statesList, actionsList, rewardsList, state_tp1sList, isGood = self.getMultipleRandomExperience(trainCount)

        # get reward tp1
        r_tp1 = list()
        for ep in range(trainCount):
            rewards = np.asarray(rewardsList[ep], dtype=theano.config.floatX)
            state_tp1s = np.asarray(state_tp1sList[ep], dtype=theano.config.floatX)

            nonTerm = np.where(rewards == 0)
            r_tp1.append(cnn.get_output(state_tp1s[nonTerm[0]]))

        if isGood:
            for ep in range(trainCount):
                states = np.asarray(statesList[ep])
                actions = np.asarray(actionsList[ep], int)
                rewards = np.asarray(rewardsList[ep], dtype=theano.config.floatX)                

                nonTerm = np.where(rewards == 0)
                rewards[nonTerm[0]] = np.max(r_tp1[ep], axis=1)*self.discount

                rewVals = np.zeros((self.miniBatch, self.numActions), dtype=theano.config.floatX)
                arange = np.arange(self.miniBatch)
                rewVals[arange, actions] = rewards

                mask = np.zeros((self.miniBatch, self.numActions), dtype=theano.config.floatX)
                nonZero = np.where(rewVals != 0)
                mask[nonZero[0], nonZero[1]] = 1
                cost += cnn.train(states, rewVals, mask)

            if cost > 0:
                self.costList.append(cost/trainCount)

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
        one_hot = np.zeros((states.shape[0], self.numActions))
        one_hot[aran, actions] = rewards[non_zero_rew]
        mask = np.zeros(one_hot.shape)
        nonZ = np.where(one_hot != 0)
        mask[nonZ[0], nonZ[1]] = 1
        loss = np.mean(mask*(output-one_hot)**2)

        return loss


import unittest


class TestExperienceHandler(unittest.TestCase):
    def test_validation(self):
        expHandler = self.getExpHandler(100)
        class dummy:
            def get_output(self, x):
                return np.random.randn(x.shape[0], 7)
        expHandler.validation(dummy())

    def getExpHandler(self, numDataToGet):
        expHandler = ExperienceHandler(32, 0.95, 1000, 7)
        for i in range(numDataToGet):
            states = np.random.random((4, 105, 80))
            rewards = np.random.randint(0, 2)
            actions = np.random.randint(0, 7)
            expHandler.addExperience(states, actions, rewards)
        return expHandler

if __name__ == '__main__':
    unittest.main()