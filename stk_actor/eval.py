# eval.py
import torch
import gymnasium as gym
from stable_baselines3 import TD3
from pystk2_gymnasium import AgentSpec
from stk_actor.pystk_actor import get_wrappers, env_name, player_name
from functools import partial
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

NUM_EPISODES = 5

def main():
    base_env = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode="human",
        difficulty =0,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env = base_env()

    model_path = "./models/model.zip"
    model = TD3.load(model_path)
    print(env)

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            env.render()

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
