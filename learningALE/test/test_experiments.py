import pytest

@pytest.mark.slow
def test_dqn_nips():
    import experiments.reproduction.DQN_NIPS.breakout_dqn
    experiments.reproduction.DQN_NIPS.breakout_dqn.main(5)
