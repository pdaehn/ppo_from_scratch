import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from ppo.ppo import PPO


class DummyLogger:
    """
    Collects calls without writing to disk.
    """

    def __init__(self):
        self.scalars = {}
        self.hparams = {}

    def log_hyperparams(self, cfg):
        self.hparams = cfg

    def log_scalars(self, scalars: dict, step: int, prefix: str = ""):
        # record last call
        self.scalars.update({f"{prefix}/{k}": v for k, v in scalars.items()})

    def log_scalar(self, tag: str, value: float, step: int):
        self.scalars[tag] = value


def make_simple_envs(num_envs: int) -> SyncVectorEnv:
    """
    Build a SyncVectorEnv of CartPole-v1 for testing.
    """

    return SyncVectorEnv(
        [lambda: __import__("gymnasium").make("CartPole-v1") for _ in range(num_envs)]
    )


def make_config(num_envs: int, rollout_length: int):
    return {
        "ppo": {
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "clip_epsilon": 0.2,
            "value_clip": True,
            "norm_adv": False,
        },
        "training": {
            "num_envs": num_envs,
            "rollout_length": rollout_length,
            "minibatch_size": num_envs * rollout_length,
            "update_epochs": 1,
            "lr": 1e-2,
            "anneal_lr": False,
            "max_steps": rollout_length * num_envs * 1,  # single epoch
        },
        "model": {
            "trunk_hidden": [16, 16],
            "actor_hidden": [],
            "critic_hidden": [],
            "activation": "tanh",
            "output_activation": None,
        },
    }


def test_smoke_train_and_eval():
    num_envs = 2
    rollout_length = 4

    envs = AsyncVectorEnv(
        [lambda: __import__("gymnasium").make("CartPole-v1") for _ in range(num_envs)]
    )

    logger = DummyLogger()

    cfg = make_config(num_envs, rollout_length)

    ppo = PPO(
        obs_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        logger=logger,
        **cfg,
    )

    ppo.collect_rollouts(envs)
    assert ppo.buffer.ptr == rollout_length, "Buffer pointer did not advance correctly"

    ppo.update()
    assert any("policy_loss" in k for k in logger.scalars), "No losses logged"
    for v in logger.scalars.values():
        assert np.isfinite(v), f"Logged scalar {v} is not finite"

    mean_r = ppo.rollout_mean_reward(envs)
    assert isinstance(mean_r, float)
    assert np.isfinite(mean_r), "Mean reward is not finite"
