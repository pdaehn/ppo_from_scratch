from pathlib import Path

import numpy as np
import torch
from gymnasium import Env, Space
from gymnasium.spaces import Discrete
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch.optim.lr_scheduler import LambdaLR

from ppo.buffer import RolloutBuffer
from ppo.model import ActorCritic
from utils.logger import TensorBoardLogger


class PPO:
    """
    Proximal Policy Optimization (PPO) agent.

    This class implements the PPO algorithm for training
     an agent in a reinforcement learning environment.
    It collects rollouts, updates the policy and value networks,
     evaluates the agent, and handles model saving/loading.
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        logger: TensorBoardLogger,
        **cfg,
    ) -> None:
        """
        Initialise the PPO agent.

        Args:
            obs_space: observation space of the environment.
            action_space: action space of the environment.
            cfg: configuration dictionary containing hyperparameters and model settings.
        """
        self.cfg = cfg
        self.global_step = 0
        self.logger = logger

        self.clip_epsilon = cfg["ppo"]["clip_epsilon"]
        self.value_clipping = cfg["ppo"]["value_clip"]
        self.vf_coef = cfg["ppo"]["vf_coef"]
        self.ent_coef = cfg["ppo"]["ent_coef"]
        self.norm_adv = cfg["ppo"]["norm_adv"]

        self.update_epochs = cfg["training"]["update_epochs"]
        self.mini_batch_size = cfg["training"]["minibatch_size"]
        self.rollout_length = cfg["training"]["rollout_length"]
        self.num_envs = cfg["training"]["num_envs"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.discrete_act_space = isinstance(action_space, Discrete)

        if self.discrete_act_space:
            self.action_dim = action_space.n
        else:
            self.action_dim = action_space.shape[0]

        self.ent_coef = (
            self.ent_coef / self.action_dim
        )  # normalize entropy coefficient by action dimension

        self.actor_critic = ActorCritic(obs_space, action_space, **cfg["model"]).to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            lr=cfg["training"]["lr"], params=self.actor_critic.parameters()
        )

        if self.cfg["training"]["anneal_lr"]:
            total_updates = self.cfg["training"]["max_steps"] // (
                self.cfg["training"]["num_envs"]
                * self.cfg["training"]["rollout_length"]
            )

            self.lr_scheduler = LambdaLR(
                self.optimizer, lr_lambda=lambda update: 1 - update / total_updates
            )

        self.buffer = RolloutBuffer(
            obs_space,
            action_space,
            cfg["ppo"]["gamma"],
            cfg["ppo"]["gae_lambda"],
            self.num_envs,
            self.rollout_length,
            self.device,
        )

    def collect_rollouts(self, envs: AsyncVectorEnv) -> None:
        """
        Collect rollouts from the environment and store them in the buffer.

        Args:
            envs: the environment to collect rollouts from.
        """
        obs, _ = envs.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

        for _ in range(self.rollout_length):
            with torch.no_grad():
                action, logp, entropy, value = self.actor_critic.act(obs)

            next_obs, reward, termination, truncation, _ = envs.step(
                action.cpu().numpy()
            )
            next_done = torch.tensor(
                np.logical_or(termination, truncation),
                dtype=torch.float32,
                device=self.device,
            )

            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

            self.buffer.add_transition(obs, action, logp, reward, next_done, value)

            obs = next_obs

        self.global_step += self.rollout_length * self.num_envs

    def update(self) -> None:
        """
        Update the policy and value networks using the collected rollouts.
        """

        with torch.no_grad():
            final_obs = self.buffer.obs_buf[-1]
            final_vals = self.actor_critic.value(final_obs)

        self.buffer.compute_gae(final_vals)

        pg_loss, vf_loss, ent_loss, clip_fracs, epoch_kl, total_loss = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for _ in range(self.update_epochs):
            ind = torch.randperm(
                self.num_envs * self.rollout_length, device=self.device
            )

            obs_b = self.buffer.obs_buf.view(-1, *self.buffer.obs_shape[2:])
            act_b = self.buffer.act_buf.view(-1, *self.buffer.act_shape[2:])
            logp_b = self.buffer.logp_buf.view(-1)
            adv_b = self.buffer.adv_buf.view(-1)
            ret_b = self.buffer.ret_buf.view(-1)
            old_value_b = self.buffer.val_buf.view(-1)

            adv_b_mean = torch.mean(adv_b)
            adv_b_std = torch.std(adv_b)

            for start in range(0, len(ind), self.mini_batch_size):
                mb = ind[start : start + self.mini_batch_size]

                if self.norm_adv:
                    adv_mb = (adv_b[mb] - adv_b_mean) / (adv_b_std + 1e-8)
                else:
                    adv_mb = adv_b[mb]

                _, new_logp, entropy, new_value = self.actor_critic.act(
                    obs_b[mb], act_b[mb]
                )

                ratio = (new_logp - logp_b[mb]).exp()

                # surrogate policy loss
                surr1 = ratio * adv_mb
                surr2 = (
                    ratio.clamp(1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_mb
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss

                if self.value_clipping:
                    old_val_mb = old_value_b[mb]
                    v_clipped = old_val_mb + (new_value - old_val_mb).clamp(
                        -self.clip_epsilon, +self.clip_epsilon
                    )

                    unclipped_vf_loss = (new_value - ret_b[mb]).pow(2)
                    clipped_vf_loss = (v_clipped - ret_b[mb]).pow(2)

                    value_loss = torch.max(unclipped_vf_loss, clipped_vf_loss).mean()
                else:
                    value_loss = (new_value - ret_b[mb]).pow(2).mean()

                # entropy loss
                entropy_loss = -entropy.mean()

                # total loss
                loss = (
                    policy_loss
                    + value_loss * self.vf_coef
                    + entropy_loss * self.ent_coef
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                clip_frac = ((ratio - 1).abs() > self.clip_epsilon).float().mean()

                clip_fracs.append(clip_frac.item())
                pg_loss.append(policy_loss.item())
                vf_loss.append(value_loss.item())
                ent_loss.append(entropy_loss.item())
                epoch_kl.append((logp_b[mb] - new_logp).mean().item())
                total_loss.append(loss.item())

        # log the losses
        self.logger.log_scalars(
            {
                "policy_loss": np.mean(pg_loss),
                "value_loss": np.mean(vf_loss),
                "entropy_loss": np.mean(ent_loss),
                "clip_frac": np.mean(clip_fracs),
                "epoch_kl": np.mean(epoch_kl),
                "total_loss": np.mean(total_loss),
            },
            self.global_step,
            "loss",
        )

    def eval_mean_reward(self, envs: SyncVectorEnv) -> float:
        """
        Evaluate agent in the environment for a full episode and return the mean reward.

        Args:
            envs: the environment to evaluate the agent in.

        Returns:
            mean_reward: the mean reward obtained during the evaluation.
        """
        next_obs, _ = envs.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        dones = np.zeros(next_obs.size(0), dtype=np.float32)
        rewards = np.zeros(next_obs.size(0), dtype=np.float32)

        while not dones.all():
            with torch.no_grad():
                action, *_ = self.actor_critic.act(next_obs)

            next_obs, reward, termination, truncation, _ = envs.step(
                action.cpu().numpy()
            )
            next_dones = np.logical_or(termination, truncation)

            rewards += reward * (1 - dones)
            dones = np.logical_or(dones, next_dones)
            next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)

        mean_reward = float(np.mean(rewards))
        self.logger.log_scalar("eval/mean_reward", mean_reward, self.global_step)
        return mean_reward

    def save(self, path: Path, model_name: str | None = None) -> None:
        """
        Save the model to the specified path.

        Args:
            path: The directory where the model will be saved.
            model_name: The name of the model file. If None, a name will be generated.
        """
        path.mkdir(parents=True, exist_ok=True)

        saved_count = len(list(path.iterdir()))
        model_name = (
            f"ppo_model_{saved_count}.pth" if model_name is None else model_name
        )

        save_path = path / model_name

        if self.discrete_act_space:
            torch.save(
                {
                    "actor": self.actor_critic.actor.state_dict(),
                    "critic": self.actor_critic.critic.state_dict(),
                    "trunk": self.actor_critic.trunk.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.cfg,
                },
                save_path,
            )
        else:
            torch.save(
                {
                    "mean_layer": self.actor_critic.mean_layer.state_dict(),
                    "log_param": self.actor_critic.log_std.detach().cpu(),
                    "trunk": self.actor_critic.trunk.state_dict(),
                    "critic": self.actor_critic.critic.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "config": self.cfg,
                },
                save_path,
            )

    def load(self, path: Path) -> None:
        """
        Load the latest model from the specified path.

        Args:
            path: The directory where the model is saved.
        """

        if not path.exists():
            raise FileNotFoundError(f"Model path {path} does not exist")

        if not any(path.iterdir()):
            print("No models found in the directory, starting from scratch.")
            return

        model_files = [p for p in path.iterdir() if p.is_file() and p.suffix == ".pth"]
        model_files.sort(key=lambda p: int(p.stem.split("_")[-1]))

        path = path / model_files[-1]

        checkpoint = torch.load(path, map_location=self.device)

        if self.discrete_act_space:
            self.actor_critic.actor.load_state_dict(checkpoint["actor"])
            self.actor_critic.critic.load_state_dict(checkpoint["critic"])
            self.actor_critic.trunk.load_state_dict(checkpoint["trunk"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.cfg = checkpoint["config"]

        else:
            self.actor_critic.mean_layer.load_state_dict(checkpoint["mean_layer"])
            self.actor_critic.log_std.data = (
                checkpoint["log_param"]
                .detach()
                .clone()
                .requires_grad_(True)
                .to(self.device)
            )
            self.actor_critic.trunk.load_state_dict(checkpoint["trunk"])
            self.actor_critic.critic.load_state_dict(checkpoint["critic"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.cfg = checkpoint["config"]

    def rollout_gif(self, env: Env) -> list[np.ndarray]:
        """
        Generate GIF frames of the agent's rollout in the environment.

        Args:
            env: the environment to generate the GIF from.
        """
        frames = []
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        done = False

        while not done:
            with torch.no_grad():
                action, *_ = self.actor_critic.act(obs)

            obs_np, _, term, trunc, _ = env.step(action.cpu().numpy())
            frame = env.render()
            frames.append(frame)

            done = bool(term or trunc)
            obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device)

        env.close()

        return frames
