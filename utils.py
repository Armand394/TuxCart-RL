import numpy as np

def preprocess_obs(obs, discrete_sizes):
    """
    obs["continuous"] : np.array
    obs["discrete"]   : np.array of ints
    discrete_sizes     : array like MultiDiscrete([...])
    """

    cont = obs["continuous"].astype(np.float32)

    # One-hot encode each discrete component
    one_hots = []
    for i, size in enumerate(discrete_sizes):
        vec = np.zeros(size, dtype=np.float32)
        vec[obs["discrete"][i]] = 1.0
        one_hots.append(vec)

    disc = np.concatenate(one_hots, axis=0)

    return np.concatenate([cont, disc], axis=0)