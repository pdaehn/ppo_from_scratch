import torch
from gymnasium.spaces import Discrete, Box
from torch import Tensor
from torch.cuda import device


class RolloutBuffer:
    """
    A buffer to store rollouts for PPO and compute GAE.

    This buffer is used to store the observations, actions, rewards, values,
    log probabilities, and done flags for a batch of rollouts.
    """

    def __init__(
        self,
        obs_space: Box | Discrete,
        action_space: Box | Discrete,
        gamma: float,
        gae_lambda: float,
        num_envs: int,
        rollout_length: int,
        device: device = torch.device("cpu"),
    ):
        """
        Initialise the rollout buffer.

        Args:
            obs_space: observation space of the environment.
            action_space: action space of the environment.
            gamma: discount factor.
            gae_lambda: lambda for GAE.
            num_envs: number of parallel environments.
            rollout_length: length of the rollout.
            device: device to store the buffer on (default: cpu).
        """
        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        B, T = self.num_envs, self.rollout_length

        self.obs_shape = (T, B, *obs_space.shape)
        self.act_shape = (T, B, *action_space.shape)

        self.obs_buf = torch.zeros(self.obs_shape, device=self.device)
        self.act_buf = torch.zeros(self.act_shape, device=self.device)
        self.logp_buf = torch.zeros((T, B), device=self.device)
        self.rew_buf = torch.zeros((T, B), device=self.device)
        self.val_buf = torch.zeros((T, B), device=self.device)
        self.done_buf = torch.zeros((T, B), device=self.device)
        self.ret_buf = torch.zeros((T, B), device=self.device)
        self.adv_buf = torch.zeros((T, B), device=self.device)

        self.ptr = 0

    def add_transition(
        self,
        obs: Tensor,
        action: Tensor,
        logp: Tensor,
        reward: Tensor,
        done: Tensor,
        value: Tensor,
    ) -> None:
        """
        Add a transition to the buffer.

        Args:
            obs: observation from the environment.
            action: action taken by the agent.
            logp: log probability of the action taken.
            reward: reward received from the environment.
            done: done flag indicating if the episode has ended.
            value: value estimate from the critic.
        """
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.logp_buf[self.ptr] = logp
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = value
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def reset_buffer(self) -> None:
        """
        Reset the buffer to its initial state.
        """
        self.obs_buf = torch.zeros_like(self.obs_buf, device=self.device)
        self.act_buf = torch.zeros_like(self.act_buf, device=self.device)
        self.logp_buf = torch.zeros_like(self.logp_buf, device=self.device)
        self.rew_buf = torch.zeros_like(self.rew_buf, device=self.device)
        self.val_buf = torch.zeros_like(self.val_buf, device=self.device)
        self.done_buf = torch.zeros_like(self.done_buf, device=self.device)
        self.ret_buf = torch.zeros_like(self.ret_buf, device=self.device)
        self.adv_buf = torch.zeros_like(self.adv_buf, device=self.device)

        self.ptr = 0

    def compute_gae(self, final_vals: Tensor) -> None:
        """
        Compute the Generalized Advantage Estimation (GAE) for the buffer.
        This is done in a backward pass through the buffer.

        Args:
              final_vals: final value estimates, for the last observation, which is not in the buffer.
        """
        B, T = self.num_envs, self.rollout_length

        lastgaelam = torch.zeros(B, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                nextnonterminal = 1.0
                nextvalues = final_vals
            else:
                nextnonterminal = 1.0 - self.done_buf[t + 1]
                nextvalues = self.val_buf[t + 1]

            delta = (
                self.rew_buf[t]
                + self.gamma * nextvalues * nextnonterminal
                - self.val_buf[t]
            )
            lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )
            self.adv_buf[t] = lastgaelam

        self.ret_buf = self.adv_buf + self.val_buf
