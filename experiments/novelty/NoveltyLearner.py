from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
from learningALE.handlers.dataset import DataSet
from learningALE.learners.nns import CNN
import numpy as np


class NoveltyLearner():
    def __init__(self, skip_frame, num_actions):
        rand_vals = (1, 0.1, 1000000)  # starting at 1 anneal eGreedy policy to 0.1 over 1,000,000 actions
        self.action_handler = ActionHandler(ActionPolicy.eGreedy, rand_vals)

        self.minimum_replay_size = 100
        self.exp_handler = DataSet(84, 84, np.random.RandomState(), max_steps=1000000, phi_length=skip_frame)
        self.cnn = CNN((None, skip_frame, 84, 84), num_actions)

        self.skip_frame = skip_frame
        self.costList = list()
        self.state_tm1 = None

        # novelty setup
        self.frame_table = dict()
        self.new_novel_states = 0

    def frames_processed(self, frames, action_performed, reward):
        # novelty reward
        for frame in frames:
            frame[frame > 0] = 1
            frame_hash = hash(frame.data.tobytes())

            # if already in table
            if frame_hash in self.frame_table:
                novelty_reward = 0
                self.frame_table[frame_hash] += 1
            # new state
            else:
                novelty_reward = 1
                self.frame_table[frame_hash] = 1
                self.new_novel_states += 1

        # if no reward from the game reward from novelty
        if reward == 0:
            reward = novelty_reward

        game_action = self.action_handler.game_action_to_action_ind(action_performed)
        if self.state_tm1 is not None:
            self.exp_handler.add_sample(self.state_tm1, game_action, reward, False)

        # generate minibatch data
        if self.exp_handler.size > self.minimum_replay_size:
            states, actions, rewards, state_tp1s, terminal = self.exp_handler.random_batch(32)
            cost = self.cnn.train(states, actions, rewards, state_tp1s, terminal)
            self.costList.append(cost)
            self.action_handler.anneal()

        self.state_tm1 = frames[-1]

    def set_legal_actions(self, legal_actions):
        self.num_actions = len(legal_actions)
        self.action_handler.set_legal_actions(legal_actions)

    def get_action(self, processed_screens):
        return self.cnn.get_output(processed_screens)[0]

    def get_game_action(self):
        return self.action_handler.action_vect_to_game_action(
            self.get_action(self.exp_handler.phi(self.state_tm1).reshape(1, self.skip_frame, 84, 84)))

    def game_over(self):
        self.exp_handler.add_terminal()  # adds a terminal
        # print('novel states', self.new_novel_states, 'total states', len(self.frame_table))
        self.new_novel_states = 0

    def get_cost_list(self):
        return self.costList

    def save(self, file):
        self.cnn.save(file)