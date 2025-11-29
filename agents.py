from torch import nn
import torch
import enum


class ActionSpace(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1
    HYBRID = 2


class Actor(nn.Module):
    """
    PPO Actor network supporting:
    - Continuous actions (Gaussian)
    - Discrete actions (Categorical)
    - Hybrid actions (continuous + multi-discrete)
    """

    def __init__(
        self,
        state_dim: int,
        action_space: ActionSpace,
        hidden_layers: list[int],
        action_dim: int = None,
        hybrid_params: dict = None
    ):
        super().__init__()

        if len(hidden_layers) != 2:
            raise ValueError("Hidden layers must be a list of two ints.")

        h1, h2 = hidden_layers

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.Tanh(),
            nn.Linear(h1, h2),
            nn.Tanh(),
        )

        self.action_space_type = action_space

        # Continuous (Gaussian)
        if action_space == ActionSpace.CONTINUOUS:
            if action_dim is None:
                raise ValueError("action_dim required for continuous actions.")

            self.cont_mu = nn.Linear(h2, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Discrete (Categorical)
        elif action_space == ActionSpace.DISCRETE:
            if action_dim is None:
                raise ValueError("action_dim required for discrete actions.")

            self.disc_logits = nn.Linear(h2, action_dim)

        # Hybrid (Gaussian + MultiDiscrete)
        elif action_space == ActionSpace.HYBRID:

            if hybrid_params is None:
                raise ValueError("hybrid_params required for hybrid action space.")

            self.cont_dim = hybrid_params["continuous_dim"]
            self.discrete_branches = hybrid_params["discrete_branches"]

            self.cont_mu = nn.Linear(h2, self.cont_dim)
            self.log_std = nn.Parameter(torch.zeros(self.cont_dim))

            # One linear head per discrete branch
            self.disc_heads = nn.ModuleList([
                nn.Linear(h2, n) for n in self.discrete_branches
            ])

        else:
            raise ValueError("Unknown action space type.")


    # Forward pass: returns raw values (mu/std or logits)
    def forward(self, state):
        z = self.shared(state)

        if self.action_space_type == ActionSpace.CONTINUOUS:
            mu = torch.tanh(self.cont_mu(z))
            std = self.log_std.exp()
            return mu, std

        elif self.action_space_type == ActionSpace.DISCRETE:
            logits = self.disc_logits(z)
            return logits

        elif self.action_space_type == ActionSpace.HYBRID:
            mu = torch.tanh(self.cont_mu(z))
            std = self.log_std.exp()
            logits = [head(z) for head in self.disc_heads]
            return mu, std, logits

        else:
            raise ValueError("Invalid action space type.")
