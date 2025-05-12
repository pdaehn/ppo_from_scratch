import torch
import torch.nn as nn
from gymnasium.spaces import Discrete
from torch.distributions import (
    Categorical,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

from utils.model_utils import build_mlp


class ActorCritic(nn.Module):
    """
    A class to build a shared trunk actor-critic model.
    The model consists of a shared trunk, an actor head, and a critic head.
    The actor head is used for discrete or continuous action spaces.
    The critic head is used for value estimation.
    """

    def __init__(
        self,
        obs_space,
        action_space,
        trunk_hidden: list[int],
        actor_hidden: list[int] | None = None,
        critic_hidden: list[int] | None = None,
        activation: str = "tanh",
        output_activation: str | None = None,
    ):
        """
        Initialise the actor-critic model.

        Args:
            obs_space: observation space of the environment.
            action_space: action space of the environment.
            trunk_hidden: hidden sizes for the shared trunk.
            actor_hidden: hidden sizes for the actor head (default: None).
            critic_hidden: hidden sizes for the critic head (default: None).
            activation: activation function for the shared trunk (default: "tanh").
            output_activation: activation function for the output layer (default: None).
        """
        super().__init__()
        obs_dim = obs_space.shape[0]
        self.discrete = isinstance(action_space, Discrete)
        self.action_dim = action_space.n if self.discrete else action_space.shape[0]

        # Build a shared trunk
        self.trunk = build_mlp(
            input_dim=obs_dim,
            output_dim=trunk_hidden[-1],
            hidden_sizes=trunk_hidden[:-1],
            activation=activation,
        )

        head_hidden = actor_hidden or []
        if self.discrete:

            self.actor = build_mlp(
                input_dim=trunk_hidden[-1],
                output_dim=self.action_dim,
                hidden_sizes=head_hidden,
                activation=activation,
                output_activation=output_activation,  # usually None
            )
        else:

            self.mean_layer = build_mlp(
                input_dim=trunk_hidden[-1],
                output_dim=self.action_dim,
                hidden_sizes=head_hidden,
                activation=activation,
            )

            self.log_std = nn.Parameter(torch.zeros(self.action_dim))
            self.act_low = torch.as_tensor(action_space.low, dtype=torch.float32)
            self.act_high = torch.as_tensor(action_space.high, dtype=torch.float32)

        head_hidden = critic_hidden or []
        self.critic = build_mlp(
            input_dim=trunk_hidden[-1],
            output_dim=1,
            hidden_sizes=head_hidden,
            activation=activation,
        )

    def act(self, obs, action=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the action, log-probability, entropy, and value for a given observation.

        Args:
            obs: observation from the environment.
            action: action to be taken (default: None).

        Returns:
            a: action to be taken.
            logp: log probability of the action.
            ent: entropy of the action distribution.
            value: value estimate from the critic.
        """


        features = self.trunk(obs)

        if self.discrete:
            logits = self.actor(features)
            dist = Categorical(logits=logits)
            a = dist.sample() if action is None else action
            logp = dist.log_prob(a)
            ent = dist.entropy()

        else:
            mean = self.mean_layer(features)
            std = self.log_std.exp().expand_as(mean)
            base = Normal(mean, std)
            dist = TransformedDistribution(base, [TanhTransform()])

            if action is None:
                raw_a = dist.rsample()  # in range (−1,1)
            else:
                # `action` is the env-scaled action in [low,high]
                # invert the affine rescale to get back into (−1,1)
                low, high = self.act_low.to(obs), self.act_high.to(obs)
                raw_a = 2 * (action - low) / (high - low) - 1
                raw_a = raw_a.clamp(-0.999, +0.999)

            logp = dist.log_prob(raw_a).sum(-1)
            ent = base.entropy().sum(-1)    # entropy of the base distribution, not implemented in the transform

            low, high = self.act_low.to(obs), self.act_high.to(obs)
            a = low + (raw_a + 1) * 0.5 * (high - low)

        value = self.critic(features).squeeze(-1)
        return a, logp, ent, value

    def value(self, obs) -> torch.Tensor:
        """
        Compute the value estimate from the critic head.

        Args:
            obs: observation from the environment.
        Returns:
            value: value estimate from the critic.
        """
        return self.critic(self.trunk(obs)).squeeze(-1)
