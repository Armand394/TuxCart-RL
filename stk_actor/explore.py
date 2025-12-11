import numpy as np
from stable_baselines3 import SAC
from tqdm import tqdm
# learn.py

import matplotlib.pyplot as plt
from functools import partial

from pystk2_gymnasium import AgentSpec


from bbrl.agents.gymnasium import make_env

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .pystk_actor import env_name, get_wrappers, player_name,get_wrappers_train


base_env = partial(
    make_env,
    env_name,
    wrappers=get_wrappers_train(),
    render_mode=None,
    difficulty =0,
    agent=AgentSpec(use_ai=False, name=player_name),
)

env = DummyVecEnv([
    lambda: Monitor(base_env())
])
env = VecNormalize(env, norm_obs=True, norm_reward=False)
path = "stk_actor/models/model.zip"
model = SAC.load(path)

def evaluate_policy(env, model, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)

def permutation_importance(env, model, n_samples=200):
    obs_dim = env.observation_space.shape[0]


    observations = []
    obs = env.reset()
    for _ in range(n_samples):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated = env.step(action)
        if terminated or truncated:
            obs= env.reset()
        observations.append(obs.copy())

    observations = np.array(observations)

    baseline = evaluate_policy(env, model)
    print("Baseline performance:", baseline)

    importances = []

    for i in tqdm(range(obs_dim)):
        obs_perturbed = observations.copy()
        obs_perturbed = obs_perturbed.reshape(-1, obs_dim)
        np.random.shuffle(obs_perturbed[:, i])

        diff_rewards = []
        for obs in obs_perturbed[:50]:
            action0, _ = model.predict(obs, deterministic=True)
            diff_rewards.append(1)  

        score = evaluate_policy(env, model)

        importances.append(baseline - score)

    return np.array(importances)


importances = permutation_importance(env, model)


label_names = [

"powerup",
"max_steer_angle",
"energy",
"skeed_factor",
"jumping",
"distance_down_track",
"velocity_z",
"velocity_x",
"velocity_y",
"front_z",
"front_x",
"front_y",
"center_path_distance",
"center_path_z",
"center_path_x",
"center_path_y",
"paths_distance_0_0",
"paths_distance_0_0",
"paths_distance_1_0",
"paths_distance_1_1",
"paths_width_0",
"paths_width_1",
"paths_start_0_z",
"paths_start_0_x",
"paths_start_0_y",
"paths_start_1_z",
"paths_start_1_x",
"paths_start_1_y",
"paths_end_0_z",
"paths_end_0_x",
"paths_end_0_y",
"paths_end_1_z",
"paths_end_1_x",
"paths_end_1_y",]

plt.bar(range(len(importances)), importances, tick_label=label_names)
plt.xticks(rotation=90)
plt.xlabel("Feature index")
plt.ylabel("Importance (reward drop)")
plt.show()
