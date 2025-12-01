# actors.py
import torch.nn as nn
import torch
from bbrl.agents import Agent

class Actor(nn.Module):
    """
    Actor compatible with SB3 TD3 actor architecture.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.mu = nn.Sequential(
            nn.Linear(obs_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_dim),
            nn.Tanh()
        )

    # def forward(self, x, t=None):
    #     return self.mu(x)
    def forward(self, ws, t=None):
        x = ws['env/env_obs/observation']
        return self.mu(x)



class SB3Actor(Agent):
    def __init__(self, sb3_policy, deterministic=False):
        super().__init__()
        self.sb3_policy = sb3_policy
        self.deterministic = deterministic

    def forward(self, t, **kwargs):
        obs_d = self.get(("env/env_obs/discrete", t))
        obs_c = self.get(("env/env_obs/continuous", t))

        obs = {
            'discrete': obs_d,
            'continuous': obs_c
        }

        actions, _ = self.sb3_policy.predict(obs, deterministic=self.deterministic)

        actions = torch.tensor(actions, dtype=torch.float32)

        self.set(("action", t), actions)

