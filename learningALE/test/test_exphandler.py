import pytest
from learningALE.handlers.experiencehandler import ExperienceHandler
import numpy as np


@pytest.fixture(scope='module')
def exp_handler():
    exp = ExperienceHandler(5)
    return exp


def test_add_experiennce(exp_handler: ExperienceHandler):
    for add in range(10):
        state = np.ones((2, 10, 10)) * add
        action = add
        reward = add
        exp_handler.add_experience(state, action, reward)

    # test size should be 10
    assert exp_handler.num_inserted == 10

    for asser in range(10):
        state = np.ones((2, 10, 10)) * asser
        action = asser
        reward = asser
        assert np.all(exp_handler.states[asser] == state)
        assert exp_handler.actions[asser] == action
        assert exp_handler.rewards[asser] == reward

    # terminal set should have nothing in it
    assert len(exp_handler.term_states) == 0


def test_add_terminal(exp_handler: ExperienceHandler):
    state_size = len(exp_handler.states)
    exp_handler.add_terminal()
    assert state_size-1 in exp_handler.term_states


def test_get_random_experience(exp_handler: ExperienceHandler):
    assert len(list(exp_handler.term_states)) > 0  # assert there is a terminal state

    states, actions, rewards, states_tp1, terminal = exp_handler.get_random_experience(999)
    # experience handler should return nothing when num_requested > num_inserted
    assert states is None

    states, actions, rewards, states_tp1, terminal = exp_handler.get_random_experience(10)
    assert states is not None
    # we have added a terminal state so last states_tp1 should be zeros
    assert np.all(states_tp1[np.where(terminal)[0]] == np.zeros(states_tp1[0].shape))
    assert np.sum(terminal) == 1

    # make sure we got all actions/rewards
    assert np.sum(actions) == 45  # 45 = 1+2+3+4+5+6+7+8+9
    assert np.sum(rewards) == 45  # 45 = 1+2+3+4+5+6+7+8+9

    # make sure we got all states
    assert np.sum(states[:, 0, 0, 0]) == 45

    # make sure all state_tp1s are correct
    for ind in range(states.shape[0]):
        if not terminal[ind]:
            assert np.all(states[ind]+1 == states_tp1[ind])


def test_trim(exp_handler: ExperienceHandler):
    old_state_size = len(exp_handler.states)
    exp_handler.trim()

    assert exp_handler.num_inserted < old_state_size
    assert len(exp_handler.states) < old_state_size
    assert exp_handler.num_inserted == exp_handler.max_len
    assert len(exp_handler.states) == exp_handler.max_len
    assert len(exp_handler.states) == exp_handler.num_inserted
    assert exp_handler.term_states.pop() == exp_handler.num_inserted - 1


if __name__ == '__main__':
    pytest.main()
