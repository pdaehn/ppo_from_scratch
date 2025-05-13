# utils/seeding.py
import random, numpy as np, torch, gymnasium as gym


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
        self._seed = seed

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
            return self.env.reset(seed=self._seed)
        return self.env.reset(seed=seed)
