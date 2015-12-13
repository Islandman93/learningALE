import pytest
from learningALE.handlers.actionhandler import ActionHandler, ActionPolicy
import numpy as np


@pytest.fixture(scope='module')
def action_handler():
    act = ActionHandler(ActionPolicy.eGreedy, [1, 0.1, 2])
    return act


def test_set_legal_actions(action_handler: ActionHandler):
    # test to make sure action raises error on matrix input
    with pytest.raises(AssertionError):
        action_handler.set_legal_actions([[0, 2, 4, 6]])

    action_handler.set_legal_actions([0, 2, 4, 6])
    assert action_handler.numActions == 4


def test_get_action(action_handler: ActionHandler):
    action_ind = action_handler.get_action([1, 0, 0, 0], random=False)
    assert isinstance(action_ind, np.integer), "expected int got {}".format(type(action_ind))
    assert action_ind == 0

    action_handler.get_action([1, 0, 0, 0])  # just make sure random doesn't fail


def test_game_action_to_action_ind(action_handler: ActionHandler):
    action_ind = action_handler.game_action_to_action_ind(2)
    assert isinstance(action_ind, np.integer), "expected int got {}".format(type(action_ind))
    assert action_ind == 1


def test_action_vect_to_game_action(action_handler: ActionHandler):
    game_action = action_handler.action_vect_to_game_action([0, 0, 1, 0], random=False)
    assert isinstance(game_action, np.integer), "expected int got {}".format(type(game_action))
    assert game_action == 4


def test_anneal(action_handler: ActionHandler):
    action_handler.anneal()
    action_handler.anneal()
    assert action_handler.randVal == 0.1
    assert action_handler.randVal == action_handler.lowestRandVal


def test_rand_vals():
    # just test to make sure rand vals doesn't fail
    action_handler = ActionHandler(ActionPolicy.randVals, [1, 0.1, 2], [0, 2, 4, 6])
    action_handler.get_action([0, 0, 0, 0])
