__author__ = 'Ben'
import numpy as np
import theano
import handlers.experiencehandler as exph
import copy


class TDHandler(exph.ExperienceHandler):
    def __init__(self, miniBatch, discount, maxLen, numActions, bufferLen):
        super().__init__(miniBatch, discount, maxLen, numActions)
        self.bufferLen = bufferLen
        self.bufStates = list()
        self.bufRewards = list()
        self.bufActions = list()

    def addToBuffer(self, state, action, reward):
        self.bufStates.append(state)
        self.bufActions.append(action)
        self.bufRewards.append(0)

        if reward != 0:
            self.processTD(reward)

        # if len(self.bufStates) > self.bufferLen and len(self.bufRewards) > self.bufferLen and len(self.bufActions) > self.bufferLen:
        #     self.addExperience(self.bufStates[0], self.bufActions[0], self.bufRewards[0])
        #     self.bufStates.remove(self.bufStates[0])
        #     self.bufRewards.remove(self.bufRewards[0])
        #     self.bufActions.remove(self.bufActions[0])

    def processTD(self, reward):
        for revInd in range(len(self.bufRewards)-1, 0, -1):
            self.bufRewards[revInd] += reward
            reward *= self.discount

    def flushBuffer(self):
        self.states.extend(copy.copy(self.bufStates))
        self.actions.extend(copy.copy(self.bufActions))
        self.rewards.extend(copy.copy(self.bufRewards))
        self.bufStates.clear()
        self.bufRewards.clear()
        self.bufActions.clear()

        while len(self.states) > self.maxLen:
            del self.states[0]
        while len(self.actions) > self.maxLen:
            del self.actions[0]
        while len(self.rewards) > self.maxLen:
            del self.rewards[0]

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
            if nonTerm[0].size > 0:
                r_tp1 = cnn.get_output(state_tp1s[nonTerm[0]].reshape((-1,) + state_tp1s.shape[1:]))
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
        non_zero_rew = np.where(abs(rewards) >= 1)[0]
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
    def test_td(self):
        expHandler = self.getExpHandler(101)
        assert len(expHandler.bufRewards) == 0
        assert expHandler.rewards[-1] == 1
        assert expHandler.rewards[-2] == 1*expHandler.discount

    def getExpHandler(self, numDataToGet):
        expHandler = TDHandler(32, 0.95, 1000, 7, 16)
        for i in range(numDataToGet):
            states = np.random.random((4, 105, 80))
            rewards = 0
            if i % 10 == 0 and i > 0:
                rewards = 1
            actions = np.random.randint(0, 7)
            expHandler.addToBuffer(states, actions, rewards)
        return expHandler

if __name__ == '__main__':
    unittest.main()