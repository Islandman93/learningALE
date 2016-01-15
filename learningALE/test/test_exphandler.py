import pytest
from learningALE.handlers.experiencehandler import ExperienceHandler
import numpy as np


@pytest.fixture(scope='module')
def exp_handler():
    exp = ExperienceHandler(5, 2, replay_start_size=18)
    return exp


def add_experience(exp_handler: ExperienceHandler, data:int):
    state = np.ones((2, 10, 10)) * data
    action = data
    reward = data
    exp_handler.add_experience(state, action, reward)


def test_add_experiennce(exp_handler: ExperienceHandler):
    for add in range(9):
        add_experience(exp_handler, add)

    # test size should be 9*2 (we added a numpy array with 3dims and 2 on the first axis
    assert exp_handler.size == 18

    for asser in range(18):
        state = np.ones((10, 10)) * asser
        action = asser
        reward = asser
        assert np.all(exp_handler.states[asser] == state//2)
        assert exp_handler.actions[asser//exp_handler.frame_skip] == action//2
        assert exp_handler.rewards[asser//exp_handler.frame_skip] == reward//2

    # terminal set should have nothing in it
    assert len(exp_handler.term_states) == 0


def test_add_terminal(exp_handler: ExperienceHandler):
    state_size = len(exp_handler.rewards)  # terminal ind is based on action/reward ind
    exp_handler.add_terminal()
    assert state_size-1 in exp_handler.term_states


def test_get_random_experience(exp_handler: ExperienceHandler):
    assert len(list(exp_handler.term_states)) > 0  # assert there is a terminal state

    states, _, _, _, _, _ = exp_handler.get_random_experience(1)
    # experience handler should return nothing when size < replay_start_size
    assert states is None

    add_experience(exp_handler, 9)
    states, actions, rewards, states_tp1, terminal, inds = exp_handler.get_random_experience(9)
    assert states is not None
    # if we got a terminal state states_tp1 should be zeros
    assert np.all(states_tp1[np.where(terminal)[0]] == np.zeros(states_tp1[0].shape))

    # make sure we got correct actions/rewards
    assert np.sum(actions) == np.sum(inds)
    assert np.sum(rewards) == np.sum(inds)


def test_trim(exp_handler: ExperienceHandler):
    old_state_size = len(exp_handler.states)
    exp_handler.trim()

    assert exp_handler.size < old_state_size
    assert len(exp_handler.states) < old_state_size
    assert exp_handler.size <= exp_handler.max_len
    assert len(exp_handler.states) <= exp_handler.max_len
    assert len(exp_handler.states) == exp_handler.size
    assert exp_handler.term_states.pop() == len(exp_handler.rewards) - 2 #minus 2 because we added a state after terminal

    # check term_states is still a set
    assert isinstance(exp_handler.term_states, set)

    # Test that terminal states that have been deleted are removed
    # add new terminal
    exp_handler.add_terminal()
    # add new states and trim
    for add in range(10):
        state = np.ones((2, 10, 10)) * add
        action = add
        reward = add
        exp_handler.add_experience(state, action, reward)
    exp_handler.trim()

    assert len(list(exp_handler.term_states)) == 0
