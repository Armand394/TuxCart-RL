# eval.py
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from pystk2_gymnasium import AgentSpec
from stk_actor.pystk_actor import get_wrappers, env_name, player_name
from functools import partial
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from pathlib import Path
import inspect
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
    mod_path = Path(inspect.getfile(get_wrappers)).parent

    env = base_env()
    env = Monitor(env)
    env = DummyVecEnv([base_env])
    env = VecNormalize.load(mod_path / "vecnormalize.pkl", env)
    env.training = False
    env.norm_reward = False
    env.metadata["fps"] = 60
    mod_path = Path(inspect.getfile(get_wrappers)).parent

    model = SAC.load(mod_path / "models/model_fix.zip",env=env)


    for episode in range(NUM_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                action, _states = model.predict(obs, deterministic=True)
                print("Action:", action)
                
                obs, reward, terminated, truncated = env.step(action)
                done = terminated[0] or truncated[0]['TimeLimit.truncated']
                print("done:", done)
                print("terminated:", terminated)
                print("truncated:", truncated)
                total_reward += reward

            env.render()

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
