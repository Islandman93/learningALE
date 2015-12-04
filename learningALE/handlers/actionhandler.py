__author__ = 'Ben'
import numpy as np


class ActionHandler:
    """
    The :class:`ActionHandler` class takes care of the interface between the action indexes returned from ALE and a
    vector of length (num actions). It also allows two different types of stochastic selection methods.
    :class:`ActionPolicy`-eGreedy where it randomly selects an action with probability e. Or
    :class:`ActionPolicy`-randVals where it adds noise to the action vector before choosing the index of the max action.

    This class supports linear annealing of both the eGreedy probability value and the randVals scalar.

     Parameters
     ----------
     action_policy : :class:`ActionPolicy`
        Specifies whether using eGreedy or adding randVals to the action value vector

     random_values : tuple
        Specifies which values to use for the action policy
        format: (Initial random value, ending random value, number of steps to anneal over)

     actions : tuple, list, array
        Default None, should be set by gameHandler.
        The legal actions from the :class:`libs.ale_python_interface.ALEInterface`
    """
    def __init__(self, action_policy, random_values, actions=None):
        self.actionPolicy = action_policy

        self.randVal = random_values[0]
        self.lowestRandVal = random_values[1]
        lin = np.linspace(random_values[0], random_values[1], random_values[2])
        self.diff = lin[0] - lin[1]
        self.countRand = 0
        self.actionCount = 0

        self.actions = actions
        if actions is not None:
            self.numActions = len(actions)
        else:
            self.numActions = 0

    def getAction(self, action, random=True):
        if random:
            if self.actionPolicy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action
                if np.random.uniform(0, 1) <= self.randVal:
                    action = np.random.randint(self.numActions)
                    self.countRand += 1
                else:
                    action = np.where(action == np.max(action))[0][0]
            elif self.actionPolicy == ActionPolicy.randVals:
                assert np.iterable(action)
                action += np.random.randn(self.numActions) * self.randVal
                action = np.where(action == np.max(action))[0][0]
        self.actionCount += 1

        return action

    def anneal(self):
        self.randVal -= self.diff
        if self.randVal < self.lowestRandVal:
            self.randVal = self.lowestRandVal

    def set_legal_actions(self, legal_actions):
        self.actions = legal_actions
        self.numActions = len(legal_actions)

    def game_action_to_action_ind(self, action_performed):
        return np.where(action_performed == self.actions)[0]

    def action_vect_to_game_action(self, action_vect, random=True):
        return self.actions[self.getAction(action_vect, random)]


class ActionHandlerTiring(ActionHandler):
    """
    :class:`ActionHandlerTiring` is a work in progress to create an action handler that forces the learner to try new actions
    by decreasing values in the action vector that have been tried often.
    Currently not finished.

     Parameters
     ----------
     action_policy : a :class:`ActionPolicy`
        Specifies whether using eGreedy or adding randVals to the action value vector
     random_values : a tuple that specifies which values to use for the action policy
        (Initial random value, ending random value, number of steps to anneal over)
     actions : the legal actions from the :class:`libs.ale_python_interface.ALEInterface`
    """
    def __init__(self, random_values, actions):
        self.actions = actions
        self.lastAction = 0
        self.lastActionInd = 0

        self.numActions = len(actions)
        self.inhibitor = np.zeros(len(actions))

    def getAction(self, action, inhibit=True):
        if inhibit:
            # action
            action = np.where(action == np.max(action))[0][0]
            if self.actionPolicy == ActionPolicy.eGreedy:
                # egreedy policy to choose random action
                if np.random.uniform(0, 1) <= self.randVal:
                    action = np.random.randint(self.numActions)
                    self.countRand += 1
                # else:

            elif self.actionPolicy == ActionPolicy.randVals:
                assert np.iterable(action)
                action += np.random.randn(self.numActions) * self.randVal
                action = np.where(action == np.max(action))[0][0]


        return self.actions[action], action

    def game_over(self):
        """
        Resets inhibition values to 0
        """
        self.inhibitor = np.zeros(len(self.actions))


from enum import Enum
class ActionPolicy(Enum):
    """
    :class:`ActionPolicy` is an Enum used to determine which policy an action handler should use for
    random exploration.

    Currently supported are eGreedy and the addition of random values to the action vector (randVals)

    The idea behind adding random values can be found here:
    https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
    """
    eGreedy = 1
    randVals = 2
