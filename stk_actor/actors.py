# actors.py
from pyexpat import model
import torch.nn as nn
import torch
from bbrl.agents import Agent
import gymnasium as gym
import numpy as np
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
        # obs_d = self.get(("env/env_obs/discrete", t))
        obs_c = self.get(("env/env_obs/continuous", t))[0]


        actions, _ = self.sb3_policy.predict(obs_c, deterministic=self.deterministic)
        print("="*20)
        print("SB3 Actor actions:", actions)
        print("="*20)

        actions = torch.tensor(actions, dtype=torch.float32)
        self.set(("action/acc", t), torch.tensor([actions[0]]))
        self.set(("action/steer", t), torch.tensor([actions[1]]))



class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))

class ArgmaxActor(Agent):
    """Just sample the argmax action"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = 0  # argmax for discrete space
        else:
            action = np.zeros(self.action_space.shape)
        self.set(("action", t), torch.tensor(action, dtype=torch.float32))