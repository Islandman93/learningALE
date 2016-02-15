import pytest
from learningALE.handlers.prioritizedexperiencehandler import PrioritizedExperienceHandler
import numpy as np

# prioritized experience handler uses base experience handler, so just test what's different
@pytest.fixture(scope='module')
def p_exp_handler():
    p_exp = PrioritizedExperienceHandler(2, 2)
    return p_exp


def test_add_experiennce(p_exp_handler: PrioritizedExperienceHandler):
    for add in range(3):
        state = np.ones((2, 10, 10)) * add
        action = add
        reward = add
        p_exp_handler.add_experience(state, action, reward)

    assert p_exp_handler.tree.root is not None
    assert p_exp_handler.tree.root.value == np.inf
    assert p_exp_handler.tree.root.extra_vals == 0


def test_add_terminal(p_exp_handler: PrioritizedExperienceHandler):
    state_size = len(p_exp_handler.states)
    p_exp_handler.add_terminal()
    assert p_exp_handler.tree.root.right.value == np.inf
    assert p_exp_handler.tree.root.right.right.extra_vals == state_size-1


def test_get_prioritized_experience(p_exp_handler: PrioritizedExperienceHandler):
    assert len(list(p_exp_handler.term_states)) > 0  # assert there is a terminal state

    states, _, _, _, _, _ = p_exp_handler.get_prioritized_experience(999)
    # experience handler should return nothing when num_requested > size
    assert states is None

    states, actions, rewards, states_tp1, terminal, inds = p_exp_handler.get_prioritized_experience(2)
    assert states is not None
    # we should've got a terminal which would be the first element
    assert np.sum(terminal) == 1
    assert np.all(states_tp1[0] == np.zeros(states_tp1[0].shape))

    # states should be 2 then 1
    assert np.all(states[0] == 2*np.ones(states[0].shape))
    assert np.all(states[1] == np.ones(states[0].shape))

    # make sure we got correct actions/rewards
    assert np.sum(actions) == 3  # 2+1
    assert np.sum(rewards) == 3  # 2+1

    # make sure all state_tp1s are correct
    for ind in range(states.shape[0]):
        if not terminal[ind]:
            assert np.all(states[ind]+1 == states_tp1[ind])

    # reinsert
    for ind in inds:
        p_exp_handler.tree.insert(ind, ind)

    # right should be nothing because 0 has value inf
    assert p_exp_handler.tree.root.right is None


def test_trim(p_exp_handler: PrioritizedExperienceHandler):
    old_state_size = len(p_exp_handler.states)
    p_exp_handler.trim()

    assert p_exp_handler.size < old_state_size
    assert len(p_exp_handler.states) < old_state_size
    assert p_exp_handler.size == p_exp_handler.max_len
    assert len(p_exp_handler.states) == p_exp_handler.max_len
    assert len(p_exp_handler.states) == p_exp_handler.size
    assert list(p_exp_handler.term_states)[0] == p_exp_handler.size - 1

    # check tree extra_vals have been updated
    assert p_exp_handler.tree.get_size() == p_exp_handler.size
    # 0 should still be there because it will still be as inf
    assert p_exp_handler.tree.root.value == np.inf
    assert p_exp_handler.tree.root.extra_vals == 0
    # 2 should be the only one left and it's ind should now be 1
    assert p_exp_handler.tree.root.left.value == 2
    assert p_exp_handler.tree.root.left.extra_vals == 1

    # check term_states is still a set
    assert isinstance(p_exp_handler.term_states, set)

    # Test that terminal states that have been deleted are removed
    # add new states and trim
    for add in range(1):
        state = np.ones((2, 10, 10)) * add
        action = add
        reward = add
        p_exp_handler.add_experience(state, action, reward)
    p_exp_handler.trim()

    # the left value should be deleted
    assert p_exp_handler.tree.root.left is None
    # right value should be 1 because we deleted state 1 (and it used to be 2)
    assert p_exp_handler.tree.root.right.extra_vals == 1
    assert len(list(p_exp_handler.term_states)) == 0

