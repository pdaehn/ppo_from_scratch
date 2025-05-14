import torch
from gymnasium import Space
from gymnasium.spaces import Discrete, Box
from torch import Tensor


class RolloutBuffer:
    """
    A buffer to store rollouts for PPO and compute GAE.

    This buffer is used to store the observations, actions, rewards, values,
    log probabilities, and done flags for a batch of rollouts.
    """

    def __init__(
        self,
        obs_space: Space[Box | Discrete],
        action_space: Space[Box | Discrete],
        gamma: float,
        gae_lambda: float,
        num_envs: int,
        rollout_length: int,
        device: torch.device = torch.device("cpu"),
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

        Attributes (all Tensors on `self.device`):
            obs_buf:    shape (T, B, *obs_space.shape)
            act_buf:    shape (T, B, *action_space.shape)
            logp_buf:   shape (T, B)
            rew_buf:    shape (T, B)
            val_buf:    shape (T, B)
            done_buf:   shape (T, B)
            ret_buf:    shape (T, B)
            adv_buf:    shape (T, B)
        """

        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.obs_space = obs_space
        self.action_space = action_space
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        T, B = self.rollout_length, self.num_envs

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

        if self.ptr > self.rollout_length - 1:
            raise RuntimeError(
                "Rollout buffer is full. Please call compute_gae() before adding more transitions."
            )

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
        self.obs_buf.zero_()
        self.act_buf.zero_()
        self.logp_buf.zero_()
        self.rew_buf.zero_()
        self.val_buf.zero_()
        self.done_buf.zero_()
        self.adv_buf.zero_()
        self.ret_buf.zero_()

        self.ptr = 0

    def compute_gae(self, bootstrap_value: Tensor) -> None:
        """
        Fill `self.adv_buf` and `self.ret_buf` in-place.

        `bootstrap_value` is V(s_{T}) – the critic’s estimate for the first
        unseen state (i.e. the observation after the last stored transition).
        """
        T, B = self.rollout_length, self.num_envs
        advantages = torch.zeros(T, B, device=self.device)
        last_adv = torch.zeros(B, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - self.done_buf[t]
                next_value = bootstrap_value
            else:
                next_non_terminal = 1.0 - self.done_buf[t]
                next_value = self.val_buf[t + 1]

            delta = (
                self.rew_buf[t]
                + self.gamma * next_value * next_non_terminal
                - self.val_buf[t]
            )
            last_adv = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_adv
            )
            advantages[t] = last_adv

        self.adv_buf = advantages
        self.ret_buf = advantages + self.val_buf
