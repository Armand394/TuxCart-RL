import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from pystk2_gymnasium import AgentSpec


# ================================
# 1. Wrapper to flatten observations
#    and use ONLY continuous actions
# ================================
class STKContinuousWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # --- Observation space ---
        # We convert:
        # obs = {"continuous": Box, "discrete": MultiDiscrete}
        # into a single large Box
        print(env)
        cont_obs_space = env.observation_space["continuous"]
        disc_obs_space = env.observation_space["discrete"]

        # One-hot length for discrete observations
        self.disc_dims = disc_obs_space.nvec
        self.total_disc_onehot = int(np.sum(self.disc_dims))

        low = []
        high = []

        # continuous part
        low.extend(cont_obs_space.low)
        high.extend(cont_obs_space.high)

        # discrete one-hot part: all in [0,1]
        low.extend([0.0] * self.total_disc_onehot)
        high.extend([1.0] * self.total_disc_onehot)

        self.observation_space = spaces.Box(
            low=np.array(low, dtype=np.float32),
            high=np.array(high, dtype=np.float32),
            dtype=np.float32
        )

        # --- Action space ---
        # Only use the continuous part of actions
        self.action_space = env.action_space["continuous"]


    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info


    def step(self, action):
        # Build full action dict for STK:
        full_action = {
            "continuous": action,
            "discrete": np.zeros_like(self.env.action_space["discrete"])  # ignore discrete actions
        }

        obs, reward, terminated, truncated, info = self.env.step(full_action)
        return self.observation(obs), reward, terminated, truncated, info



    def observation(self, obs):
        cont = obs["continuous"]
        disc = obs["discrete"]

        # one-hot encode each discrete component
        onehots = []
        idx = 0
        for value, n in zip(disc, self.disc_dims):
            h = np.zeros(n, dtype=np.float32)
            h[int(value)] = 1.0
            onehots.append(h)
            idx += n

        onehots = np.concatenate(onehots)

        return np.concatenate([cont, onehots]).astype(np.float32)
    
