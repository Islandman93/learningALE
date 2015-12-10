from learningALE.learners.DQN import DQNLearner


class TargetDQNLearner(DQNLearner):
    def __init__(self, skip_frame, num_actions, load=None):
        super().__init__(skip_frame, num_actions, load)
        self.target_cnn = self.cnn.copy()

    def copy_new_target(self):
        self.target_cnn = self.cnn.copy()

    def frames_processed(self, frames, action_performed, reward):
        self.exp_handler.add_experience(frames, self.action_handler.game_action_to_action_ind(action_performed), reward)
        self.train_handler.train_target(self.exp_handler, self.cnn, self.target_cnn)
        self.action_handler.anneal()
