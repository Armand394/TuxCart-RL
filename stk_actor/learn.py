# learn.py

import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from .wrappers import STKContinuousWrapper
import torch
from .actors import SB3Actor
from pathlib import Path
import inspect
from .pystk_actor import env_name, get_wrappers, player_name
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from functools import partial

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, EveryNTimesteps

def main():
    # Base environment
    # base_env = gym.make(
    #     "supertuxkart/flattened-v0",
    #     agent=AgentSpec(use_ai=False),
    #     difficulty=0,
    #     render_mode=None
    # )


    base_env = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        difficulty =0,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env = base_env()

    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)

    check_env(env, warn=True)
    mod_path = Path(inspect.getfile(get_wrappers)).parent

    checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./logs/")
    event_callback = EveryNTimesteps(n_steps=1000, callback=checkpoint_on_event)
    # ========================
    # Train TD3
    # ========================

    checkpoint_path = Path(mod_path / 'check_point.zip')

    if checkpoint_path.exists():
        print('load check point')
        model = TD3.load(checkpoint_path, env)

    else:
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=200_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(2, "episode"),
            verbose=1,
            tensorboard_log="./logs/tuxCart-td3-continuous-tb/"
        )

    model.learn(total_timesteps=6_000,tb_log_name="run_1", progress_bar = True, callback=event_callback)

    policy = model.policy

    # (3) Save the actor sate
    sb3_actor = SB3Actor(model)

    torch.save(policy.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "models/model.zip")

if __name__ == "__main__":
    main()
