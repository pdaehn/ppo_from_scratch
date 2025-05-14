import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from ppo.model import ActorCritic


@pytest.mark.parametrize(
    "action_space",
    [
        Discrete(4),
        Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
    ],
)
def test_actor_critic_act_shapes_and_ranges(action_space):
    """
    Verify that ActorCritic.act:
      - returns tensors of the right shape,
      - produces valid actions within the action space,
      - log-probabilities are finite,
      - entropy is non-negative.
    """

    obs_dim = (5,) if isinstance(action_space, Discrete) else (3,)
    obs_space = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)

    model = ActorCritic(
        obs_space=obs_space,
        action_space=action_space,
        trunk_hidden=[8, 8],
    )
    model.eval()

    batch_size = 7
    obs = torch.zeros((batch_size, *obs_dim), dtype=torch.float32)

    with torch.no_grad():
        a, logp, ent, val = model.act(obs)

    assert isinstance(a, torch.Tensor) and isinstance(logp, torch.Tensor)
    assert isinstance(ent, torch.Tensor) and isinstance(val, torch.Tensor)

    assert a.shape[0] == batch_size
    assert logp.shape == (batch_size,)
    assert ent.shape == (batch_size,)
    assert val.shape == (batch_size,)

    assert torch.isfinite(val).all()

    assert (ent >= 0).all()

    assert torch.isfinite(logp).all()

    if isinstance(action_space, Discrete):
        assert a.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        assert torch.all((a >= 0) & (a < action_space.n))
    else:
        low = torch.tensor(action_space.low, dtype=torch.float32)
        high = torch.tensor(action_space.high, dtype=torch.float32)
        eps = 1e-6
        assert torch.all(a >= (low - eps))
        assert torch.all(a <= (high + eps))


def test_action_distribution_changes_with_input():
    """
    Ensure that different observations lead to different actions/logps/values.
    """
    action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    obs_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    model = ActorCritic(
        obs_space=obs_space, action_space=action_space, trunk_hidden=[4, 4]
    )
    model.eval()

    obs1 = torch.zeros((1, 2), dtype=torch.float32)
    obs2 = torch.ones((1, 2), dtype=torch.float32) * 10.0

    with torch.no_grad():
        a1, lp1, _, v1 = model.act(obs1)
        a2, lp2, _, v2 = model.act(obs2)

    assert not torch.allclose(a1.float(), a2.float()), "Actions did not change with obs"
    assert not torch.isclose(lp1, lp2), "Log-probs did not change with obs"
    assert not torch.isclose(v1, v2), "Values did not change with obs"
