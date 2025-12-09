import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        self.action_space = env.action_space["continuous"]

    def action(self, action):
        if isinstance(action, dict):
            full_action = action
        else:
            full_action = {
                "continuous": action,
                "discrete": np.zeros_like(self.env.action_space["discrete"])
            }
        return full_action

class FlattenDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.discrete_dims = env.observation_space['discrete'].nvec
        self.continuous_dim = env.observation_space['continuous'].shape[0]
        self.continuous_space = env.observation_space['continuous']
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.continuous_dim + int(np.sum(self.discrete_dims)),),
            dtype=np.float32
        )

    def observation(self, obs):
        cont = obs['continuous']
        disc = np.concatenate([np.eye(n)[obs['discrete'][i]] for i, n in enumerate(self.discrete_dims)])
        return np.concatenate([cont, disc])