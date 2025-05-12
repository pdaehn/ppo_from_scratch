# trainer/train.py
import time
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from omegaconf import OmegaConf
from tqdm import tqdm
from ppo.ppo import PPO
from utils.logger import TensorBoardLogger


def make_env(
    env_name: str, norm_rewards: bool = True, env_kwargs: dict | None = None
) -> callable:
    """
    Returns a thunk that will create and wrap an env instance.

    Args:
        env_name: name of the gym environment, e.g. "CartPole-v1"
        norm_rewards: whether to apply NormalizeReward wrapper
        env_kwargs: any kwargs to forward into gym.make()
    """
    env_kwargs = env_kwargs or {}

    def _thunk() -> gym.Env:
        """
        Create and wrap the environment.
        """
        env = gym.make(env_name, **env_kwargs)
        env = gym.wrappers.Autoreset(env)
        if norm_rewards:
            env = gym.wrappers.NormalizeReward(env)
        return env

    return _thunk


class PPOTrainer:
    """
    A class to train a PPO agent for a given config.
    """

    def __init__(
        self, config_path: str = "../configs/default.yaml", run_name: str | None = None
    ) -> None:
        """
        Initialise the PPOTrainer.

        Args:
            config_path: path to the configuration file
            run_name: name of the run, if None a combination of env name and timestamp is used
        """

        # load hyperparameters and paths
        self.cfg = OmegaConf.load(config_path)
        self.num_envs = self.cfg["training"]["num_envs"]
        self.rollout_length = self.cfg["training"]["rollout_length"]
        self.batch_size = self.num_envs * self.rollout_length
        self.num_epochs = self.cfg["training"]["max_steps"] // self.batch_size
        self.save_interval = self.cfg["training"]["save_interval"]

        # create environment(s)
        self.train_envs = AsyncVectorEnv(
            [make_env(self.cfg["env"]["name"]) for _ in range(self.num_envs)]
        )
        self.eval_envs = SyncVectorEnv(
            [
                make_env(self.cfg["env"]["name"], norm_rewards=False)
                for _ in range(self.num_envs)
            ]
        )

        # instantiate PPO agent
        self.ppo = PPO(
            obs_space=self.train_envs.single_observation_space,
            action_space=self.train_envs.single_action_space,
            **self.cfg,
        )

        # instantiate logger
        self.run_name = (
            f"{self.cfg['env']['name']}_{int(time.time())}"
            if run_name is None
            else run_name
        )
        self.log_dir = self.cfg["logging"]["log_dir"] + "/" + self.run_name
        self.model_dir = self.cfg["training"]["model_dir"] + "/" + self.run_name

        # if we are resuming a run, load the model
        if run_name is not None:
            self.ppo.load(self.model_dir)

        self.logger = TensorBoardLogger(log_dir=self.log_dir)
        self.logger.log_hyperparams(self.cfg)

    def train(self) -> None:
        """
        Train the PPO agent.
        This function runs the training loop for the specified number of epochs.
        """
        best_reward = -float("inf")
        for i in tqdm(range(self.num_epochs), desc="Epoch"):
            num_steps = i * self.num_envs * self.rollout_length

            self.anneal_lr(i)
            self.ppo.buffer.reset_buffer()
            self.ppo.collect_rollouts(self.train_envs)
            self.ppo.update(self.logger, num_steps)
            eval_mean_reward = self.ppo.full_testing_episode(
                self.eval_envs, self.logger, num_steps
            )

            if (i + 1) % self.save_interval == 0:
                self.ppo.save(self.model_dir)

            if eval_mean_reward > best_reward:
                best_reward = eval_mean_reward
                self.ppo.save(self.model_dir, model_name="best_model.pth")

    def anneal_lr(self, i) -> None:
        """
        Linearly anneal the learning rate according to the progress of training.

        Args:
            i: current epoch
        """
        for param_group in self.ppo.optimizer.param_groups:
            if self.cfg["training"]["anneal_lr"]:
                progress_ratio = (i + 1) / self.num_epochs
                param_group["lr"] = self.cfg["training"]["lr"] * (1 - progress_ratio)
            self.logger.log_scalar("lr", param_group["lr"], i)


if __name__ == "__main__":
    trainer = PPOTrainer(run_name="CartPole-v1_1747065702")
    trainer.train()
