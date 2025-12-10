import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FlattenActionSpace(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Dict)
        print("Original action space:", env.action_space)
        # self.action_space = env.action_space["continuous"]
        low_acc = env.action_space["acceleration"].low
        high_acc = env.action_space["acceleration"].high
        low_steer = env.action_space["steer"].low
        high_steer = env.action_space["steer"].high
        low = np.concatenate([low_acc, low_steer])
        high = np.concatenate([high_acc, high_steer])
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        print("Flattened action space:", self.action_space)
    def action(self, action_dict):
        acc = action_dict["acceleration"]
        steer = action_dict["steer"]
        action= {
            'continuous': np.concatenate([acc, steer]),
            'discrete': np.array([0, 0,0, 1, 0], dtype=np.int64)
        }
        return action
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
    
class FlattenFilterWrapper(gym.ObservationWrapper):
    """
    Wrapper pour SuperTuxKart full-v0 :
    - filtre un sous-ensemble d'observations
    - tronque les champs paths_* aux 2 premiers
    - aplatit tout en un vecteur float32
    """

    def __init__(self, env):
        super().__init__(env)

        # Champs à garder dans cet ordre précis
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

        # Construire l'espace d'observation flatten
        example_obs = self.observation(env.reset()[0])
        low = np.full(example_obs.shape, -np.inf, dtype=np.float32)
        high = np.full(example_obs.shape, np.inf, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def filter_value(self, key, value):
        """Filtre un champ selon les règles."""

        if key in ["paths_distance", "paths_width", "paths_start", "paths_end"]:
            # garder les deux premiers éléments
            arr = np.array(value)
            return arr[:2].flatten()

        # Convertir scalaires ou petits tableaux en numpy 1D
        arr = np.array(value).flatten()
        return arr

    def observation(self, obs):
        """Construit un vecteur 1D flatten filtré."""

        parts = []
        for key in self.keep_keys:
            value = obs[key]
            filtered = self.filter_value(key, value)
            parts.append(filtered)

        flat = np.concatenate(parts).astype(np.float32)
        return flat
    
