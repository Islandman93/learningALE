from learningALE.learners.learner import learner
from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
from learningALE.handlers.experiencehandler import ExperienceHandler
from learningALE.handlers.dataset import DataSet
from learningALE.learners.nns import CNN
import numpy as np


class DQNLearner(learner):
    def __init__(self, skip_frame, num_actions, load=None):
        super().__init__()

        rand_vals = (1, 0.1, 1000000)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000 actions
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)

        self.minimum_replay_size = 100
        self.exp_handler = DataSet(80, 86, np.random.RandomState(), max_steps=1000000, phi_length=skip_frame)
        self.cnn = CNN((None, skip_frame, 86, 80), num_actions)

        self.skip_frame = skip_frame
        self.discount = .95
        self.costList = list()

        if load is not None:
            self.cnn.load(load)

    def frames_processed(self, frames, action_performed, reward):
        game_action = self.action_handler.game_action_to_action_ind(action_performed)
        for frame in frames:
            self.exp_handler.add_sample(frame, game_action, reward, False)

        # generate minibatch data
        if self.exp_handler.size > self.minimum_replay_size:
            states, actions, rewards, state_tp1s, terminal = self.exp_handler.random_batch(32)
            cost = self.cnn.train(states, actions, rewards, state_tp1s, terminal)
            self.costList.append(cost)
            self.action_handler.anneal()

    def get_action(self, game_input):
        return self.cnn.get_output(game_input)[0]

    def game_over(self):
        self.exp_handler.add_terminal()  # adds a terminal

    def get_game_action(self, game_input):
        return self.action_handler.action_vect_to_game_action(self.get_action(game_input.reshape((1, self.skip_frame, 86, 80))))

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)

    def save(self, file):
        self.cnn.save(file)

    def get_cost_list(self):
        return self.costList


class DQNTester:
    def __init__(self, skip_frame, num_actions, load, rand_val=0.05):
        rand_vals = (rand_val, rand_val, 2)
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)
        self.cnn = CNN((None, skip_frame, 86, 80), num_actions, 1)
        self.cnn.load(load)
        self.q_vals = list()
        self.skip_frame = skip_frame

    def get_game_action(self, game_input):
        q_vals = self.cnn.get_output(game_input.reshape((1, self.skip_frame, 86, 80)))[0]
        self.q_vals.append(q_vals)
        return self.action_handler.action_vect_to_game_action(q_vals)

    def frames_processed(self, a, b, c):
        pass

    def set_legal_actions(self, legal_actions):
        self.action_handler.set_legal_actions(legal_actions)


