# trainer/train.py
import time
from pathlib import Path

import gymnasium as gym
import imageio
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from omegaconf import OmegaConf
from tqdm import tqdm

from ppo.ppo import PPO
from utils.logger import TensorBoardLogger
from utils.seeding import SeedWrapper, set_seed


def make_env(
    norm_rewards: bool = True,
    seed: int | None = 42,
    env_kwargs: dict | None = None,
) -> callable:
    """
    Returns a thunk that will create and wrap an env instance.

    Args:
        norm_rewards: whether to apply NormalizeReward wrapper
        seed: seed for the environment
        env_kwargs: any kwargs to forward into gym.make()
    """
    env_kwargs = env_kwargs or {}

    def _thunk() -> gym.Env:
        """
        Create and wrap the environment.
        """
        env = gym.make(**env_kwargs)
        env = gym.wrappers.Autoreset(env)
        env = SeedWrapper(env, seed)
        if norm_rewards:
            env = gym.wrappers.NormalizeReward(env)
        return env

    return _thunk


class PPOTrainer:
    """
    A class to train a PPO agent for a given config.
    """

    def __init__(self, config_path: Path | None = None) -> None:
        """
        Initialise the PPOTrainer.

        Args:
            config_path: path to the configuration file
        """

        if config_path is None:
            pkg_root = Path(__file__).resolve().parent.parent
            config_path = pkg_root / "configs" / "default.yaml"

        # load hyperparameters and paths
        self.cfg = OmegaConf.load(config_path)
        self.num_envs = self.cfg["training"]["num_envs"]
        self.rollout_length = self.cfg["training"]["rollout_length"]
        self.batch_size = self.num_envs * self.rollout_length
        self.num_epochs = self.cfg["training"]["max_steps"] // self.batch_size
        self.save_interval = self.cfg["training"]["save_interval"]
        self.seed = self.cfg["training"]["seed"]
        set_seed(self.seed)

        # create environment(s)
        self.train_envs = AsyncVectorEnv(
            [
                make_env(seed=self.seed + i, env_kwargs=self.cfg["env"])
                for i in range(self.num_envs)
            ]
        )
        self.eval_envs = SyncVectorEnv(
            [
                make_env(
                    norm_rewards=False, seed=self.seed + i, env_kwargs=self.cfg["env"]
                )
                for i in range(self.num_envs)
            ]
        )

        # instantiate logger
        self.run_name = f"{self.cfg['env']['id']}_{int(time.time())}"

        self.log_dir = Path(self.cfg["logging"]["log_dir"]) / self.run_name
        self.model_dir = Path(self.cfg["training"]["model_dir"]) / self.run_name
        self.gif_dir = Path(self.cfg["training"]["gif_dir"]) / self.run_name

        self.logger = TensorBoardLogger(log_dir=self.log_dir)
        self.logger.log_hyperparams(self.cfg)

        # instantiate PPO agent
        self.ppo = PPO(
            obs_space=self.train_envs.single_observation_space,
            action_space=self.train_envs.single_action_space,
            logger=self.logger,
            **self.cfg,
        )

    def train(self) -> None:
        """
        Train the PPO agent.
        This function runs the training loop for the specified number of epochs.
        """
        best_reward = -float("inf")
        for i in tqdm(range(self.num_epochs), desc="Epoch"):
            self.ppo.buffer.reset_buffer()
            self.ppo.collect_rollouts(self.train_envs)
            self.ppo.update()
            self.step_lr()

            if (i + 1) % self.save_interval == 0:
                self.ppo.save(self.model_dir)

            eval_mean_reward = self.mean_reward_eval()

            if eval_mean_reward > best_reward:
                best_reward = eval_mean_reward
                self.ppo.save(self.model_dir, model_name="best_model.pth")

    def step_lr(self):
        if self.cfg["training"]["anneal_lr"]:
            self.ppo.lr_scheduler.step()
            current_lr = self.ppo.lr_scheduler.get_last_lr()[0]
            self.logger.log_scalar("lr", current_lr, self.ppo.global_step)
        else:
            self.logger.log_scalar(
                "lr", self.cfg["training"]["lr"], self.ppo.global_step
            )

    def render_policy_eval_gif(
        self,
        fps: int = 30,
    ):
        """
        Render the policy evaluation as a GIF.
        Args:
            fps: frames per second for the GIF
        """
        output_dir = Path(self.gif_dir) / "policy_eval.gif"
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        env_kwargs = self.cfg["env"]
        env_kwargs["render_mode"] = "rgb_array"

        env_thunk = make_env(
            norm_rewards=self.cfg["env"].get("norm_rewards", True),
            seed=self.seed,
            env_kwargs=env_kwargs,
        )
        env = env_thunk()

        frames = self.ppo.rollout_gif(env)

        imageio.mimsave(output_dir, frames, format="GIF", duration=1 / fps, loop=0)

    def mean_reward_eval(self):
        """
        Evaluate the mean reward of the policy.
        """
        mean_reward = self.ppo.rollout_mean_reward(self.eval_envs)
        return mean_reward


if __name__ == "__main__":
    trainer = PPOTrainer()
    trainer.train()
    trainer.render_policy_eval_gif()
