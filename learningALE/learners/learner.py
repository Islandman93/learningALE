from abc import ABCMeta


class learner(metaclass=ABCMeta):
    def __init__(self):
        pass

    def frames_processed(self, frames, action_performed, reward):
        pass

    def get_game_action(self, game_input):
        pass

    def get_action(self, game_input):
        pass

    def set_legal_actions(self, legal_actions):
        pass

    def save(self, file):
        pass
