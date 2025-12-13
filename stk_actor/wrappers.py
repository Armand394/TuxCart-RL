import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pystk2_gymnasium.wrappers import ActionObservationWrapper
import pickle 
class FlattenActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        # self.action_space = env.action_space["continuous"]
        low_acc = env.action_space["acceleration"].low
        high_acc = env.action_space["acceleration"].high
        low_steer = env.action_space["steer"].low
        high_steer = env.action_space["steer"].high
        low = np.concatenate([low_acc, low_steer])
        high = np.concatenate([high_acc, high_steer])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def action(self, action):

        accel = float(action[0])
        steer = float(action[1])

        full_action = {
            "acceleration": np.array([accel], dtype=np.float32),
            "steer": np.array([steer], dtype=np.float32),

            # forcés à zéro
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0,
        }

        return full_action
    

class FlattenActionSpaceEval(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        # self.action_space = env.action_space["continuous"]
        low_acc = env.action_space["acceleration"].low
        high_acc = env.action_space["acceleration"].high
        low_steer = env.action_space["steer"].low
        high_steer = env.action_space["steer"].high
        low = np.concatenate([low_acc, low_steer])
        high = np.concatenate([high_acc, high_steer])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    def action(self, action):

        accel = action["acc"]
        steer = action["steer"]

        full_action = {
            "acceleration": np.array([accel], dtype=np.float32),
            "steer": np.array([steer], dtype=np.float32),

            # forcés à zéro
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0,
        }
        return full_action

class OnlySteerAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        # self.action_space = env.action_space["continuous"]
        low_steer = env.action_space["steer"].low
        high_steer = env.action_space["steer"].high
        self.action_space = spaces.Box(low=low_steer, high=high_steer, dtype=np.float32)
    def action(self, action):

        # accel = float(action[0])
        steer = float(action[0])

        full_action = {
            "acceleration": 1.0,
            "steer": np.array([steer], dtype=np.float32),

            # forcés à zéro
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0,
        }

        return full_action
    
class OnlySteerActionEval(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        # self.action_space = env.action_space["continuous"]
        low_steer = env.action_space["steer"].low
        high_steer = env.action_space["steer"].high
        self.action_space = spaces.Box(low=low_steer, high=high_steer, dtype=np.float32)
    def action(self, action):

        # accel = float(action[0])
        steer = float(action["steer"])

        full_action = {
            "acceleration": 1.0,
            "steer": np.array([steer], dtype=np.float32),

            # forcés à zéro
            "brake": 0,
            "drift": 0,
            "fire": 0,
            "nitro": 0,
            "rescue": 0,
        }

        return full_action
    
    
class FilterWrapperEval(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.keep_keys = [
            "powerup",
            "max_steer_angle",
            "energy",
            "skeed_factor",
            "jumping",
            "distance_down_track",
            "velocity",
            "front",
            "center_path_distance",
            "center_path",
            "paths_distance",
            "paths_width",
            "paths_start",
            "paths_end",
        ]


        low = np.full(34, -np.inf, dtype=np.float32)
        high = np.full(34, np.inf, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def filter_value(self, key, value):

        if key in ["paths_distance", "paths_width", "paths_start", "paths_end"]:
            # garder que les deux premiers segments
            arr = np.array(value)
            return arr[:2].flatten()

        arr = np.array(value).flatten()
        return arr

    def observation(self, obs):
        parts = []
        for key in self.keep_keys:
            value = obs[key]
            filtered = self.filter_value(key, value)

            parts.append(filtered)
            # parts[key] = filtered

        flat = np.concatenate(parts).astype(np.float32)
        
        return {"continuous": flat}
    
class FilterWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

        self.keep_keys = [
            "powerup",
            "max_steer_angle",
            "energy",
            "skeed_factor",
            "jumping",
            "distance_down_track",
            "velocity",
            "front",
            "center_path_distance",
            "center_path",
            "paths_distance",
            "paths_width",
            "paths_start",
            "paths_end",
        ]

        obs, _ = env.reset()  # env.reset()
        sample_obs = self.observation(obs)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=sample_obs.shape, dtype=np.float32
        )


    def filter_value(self, key, value):
        arr = np.array(value)

        if key in ["paths_distance"]:
            # paths_distance = (N, 2) -> garder 2 premiers -> shape (2,2)
            return arr[:2].reshape(-1)

        if key in ["paths_width"]:
            # paths_width = (N, 1) -> garder 2 premiers -> shape (2,1)
            return arr[:2].reshape(-1)

        if key in ["paths_start", "paths_end"]:
            # (N, 3) -> garder 2 -> (2,3)
            return arr[:2].reshape(-1)

        return arr.reshape(-1)


    def observation(self, obs):
        parts = []
        for key in self.keep_keys:
            parts.append(self.filter_value(key, obs[key]))
        return np.concatenate(parts).astype(np.float32)
    

class OnlyContinuousActionsWrapper(ActionObservationWrapper):
    """Removes the discrete actions"""

    def __init__(self, env: gym.Env, **kwargs):
        super().__init__(env, **kwargs)

        self.discrete_actions = spaces.Dict(
            {
                key: value
                for key, value in env.action_space.items()
                if isinstance(value, spaces.Discrete)
            }
        )

        self._action_space = spaces.Dict(
            {
                key: value
                for key, value in env.action_space.items()
                if isinstance(value, spaces.Box)
            }
        )

    def observation(self, obs):
        if "action" in obs:
            obs = {**obs}
            obs["action"] = {
                key: obs["action"][key] for key in self.action_space.keys()
            }
        return obs

    def action(self, action) :
        return {**action, **{key: 0 for key, _ in self.discrete_actions.items()}}
    


class ObsNormalizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, vecnormalize_path, epsilon=1e-8, clip_obs=10.0):
        super().__init__(env)

        with open(vecnormalize_path, "rb") as f:
            vecnorm = pickle.load(f)

        self.mean = vecnorm.obs_rms.mean
        self.var = vecnorm.obs_rms.var
        self.epsilon = epsilon
        self.clip_obs = clip_obs

        assert self.observation_space.shape == self.mean.shape, (
            "Observation space incompatible avec VecNormalize"
        )

    def observation(self, obs):
        obs = obs["continuous"]
        obs = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return {"continuous": np.clip(obs, -self.clip_obs, self.clip_obs)}