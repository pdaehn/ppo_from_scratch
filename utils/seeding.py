# utils/seeding.py
import random

import gymnasium as gym
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set the seed for random number generation in Python, NumPy, and Torch.

    Args:
        seed: the seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SeedWrapper(gym.Wrapper):
    """
    A wrapper to set a fixed seed for environments.
    """

    def __init__(self, env: gym.Env, seed: int):
        """
        Initialise the SeedWrapper.

        Args:
            env: the environment to wrap.
            seed: the seed value to set for the environment.
        """
        super().__init__(env)
        seed_sequence = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(seed_sequence)

    def reset(self, seed: int = None, **kwargs):
        """
        Reset the environment with a fixed seed.
        If no seed is provided, the environment's internal seed is used.

        Args:
            seed: the seed value to set for the environment.
            **kwargs: additional keyword arguments to pass to the reset method to match the
                signature of the original environment's reset method.
        """

        if seed is None:
            child_seed = int(self._rng.integers(0, 2**32))
            return self.env.reset(seed=child_seed)
        return self.env.reset(seed=seed)
