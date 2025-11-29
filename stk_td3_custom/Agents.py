# actors.py
import torch.nn as nn
import torch
from bbrl.agents import Agent
from torch.distributions import Normal

class Actor(nn.Module):
    """
    TD3 Actor
    """
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()

        # Observation space
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.hidden1 = kwargs.get("hidden1", 256)
        self.hidden2 = kwargs.get("hidden2", 256)
        
        self.actor_nn = nn.Sequential(
            nn.Linear(obs_dim, self.hidden1),
            nn.ReLU(),
            nn.Linear( self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, act_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.actor_nn(obs)
    

class Critic(nn.Module):
    """
    TD3 Critic
    """
    def __init__(self, observation_space, action_space, **kwargs):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.hidden1 = kwargs.get("hidden1", 256)
        self.hidden2 = kwargs.get("hidden2", 256)

        self.critic_nn = nn.Sequential(
            nn.Linear((obs_dim+act_dim), self.hidden1),
            nn.ReLU(),
            nn.Linear(self.hidden1, self.hidden2),
            nn.ReLU(),
            nn.Linear(self.hidden2, 1),
        )

    def forward(self, obs, act):
        obs_act = torch.cat((obs, act), dim=1)
        return self.critic_nn(obs_act)


class GaussianNoise(nn.Module):
    def __init__(self, sigma_start, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma =  sigma_start

    def forward(self, action):
        dist = Normal(action, self.sigma)
        action_noisy = dist.sample()
        return action_noisy

