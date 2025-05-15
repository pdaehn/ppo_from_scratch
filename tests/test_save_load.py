from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete

from ppo.ppo import PPO


class DummyLogger:
    """
    Dummy logger that does nothing, used for testing.
    """

    def log_scalars(self, *args, **kwargs):
        pass

    def log_scalar(self, *args, **kwargs):
        pass


@pytest.mark.parametrize(
    "action_space",
    [
        Discrete(3),
        Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    ],
)
def test_save_and_load_consistency(tmp_path, action_space):
    """
    Test that saving and loading a PPO model preserves actor-critic parameters.
    """
    cfg = {
        "ppo": {
            "clip_epsilon": 0.2,
            "value_clip": True,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "norm_adv": True,
        },
        "training": {
            "update_epochs": 1,
            "minibatch_size": 2,
            "rollout_length": 1,
            "num_envs": 1,
            "lr": 1e-3,
            "anneal_lr": False,
            "max_steps": 1,
        },
        "model": {"trunk_hidden": [4, 4]},
    }
    obs_dim = (4,) if isinstance(action_space, Discrete) else action_space.shape
    obs_space = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)

    ppo_orig = PPO(obs_space, action_space, DummyLogger(), **cfg)

    for param in ppo_orig.actor_critic.parameters():
        param.data.fill_(0.123)

    save_dir = tmp_path / "models"
    ppo_orig.save(save_dir)

    files = list(save_dir.iterdir())
    assert len(files) == 1

    ppo_loaded = PPO(obs_space, action_space, DummyLogger(), **cfg)

    for param in ppo_loaded.actor_critic.parameters():
        param.data.fill_(0.987)

    ppo_loaded.load(save_dir)

    orig_params = list(ppo_orig.actor_critic.parameters())
    loaded_params = list(ppo_loaded.actor_critic.parameters())
    assert len(orig_params) == len(loaded_params)
    for p_orig, p_loaded in zip(orig_params, loaded_params):
        assert torch.allclose(
            p_orig, p_loaded
        ), "Loaded parameter does not match saved one"


def test_load_nonexistent_path_raises():
    """
    Test that loading from a non-existent path raises FileNotFoundError.
    """
    dummy_dir = Path("/nonexistent/path/to/models")
    cfg = {
        "ppo": {
            "clip_epsilon": 0.2,
            "value_clip": True,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "norm_adv": True,
        },
        "training": {
            "update_epochs": 1,
            "minibatch_size": 2,
            "rollout_length": 1,
            "num_envs": 1,
            "lr": 1e-3,
            "anneal_lr": False,
            "max_steps": 1,
        },
        "model": {"trunk_hidden": [4, 4]},
    }
    obs_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
    action_space = Discrete(2)
    ppo = PPO(obs_space, action_space, DummyLogger(), **cfg)
    with pytest.raises(FileNotFoundError):
        ppo.load(dummy_dir)
