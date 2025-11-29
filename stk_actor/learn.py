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
import glob


CHECKPOINT_DIR = Path("./logs/checkpoints")

def main():
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

    checkpoint_on_event = CheckpointCallback(save_freq=1, 
                                             save_path=CHECKPOINT_DIR, 
                                             save_replay_buffer=True)
    event_callback = EveryNTimesteps(n_steps=50000, callback=checkpoint_on_event)
    # ========================
    # Train TD3
    # ========================

    checkpoints = sorted(CHECKPOINT_DIR.glob("rl_model_*_steps.zip"))

    if checkpoints:
        last_checkpoint = checkpoints[-1]
        print("Load checkpoint:", last_checkpoint)

        model = TD3.load(last_checkpoint, env)

        replay_buf_path = last_checkpoint.with_suffix("").as_posix() + "_replay_buffer.pkl"
        if Path(replay_buf_path).exists():
            model.load_replay_buffer(replay_buf_path)
            print("Load replay buffer.")
        else:
            print("replay buffer not found")

    else:
        print("start from scratch")
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

    model.learn(total_timesteps=1_000_000,tb_log_name="run_1", progress_bar = True, callback=event_callback)

    policy = model.policy

    # (3) Save the actor sate
    sb3_actor = SB3Actor(model)

    torch.save(policy.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "models/model.zip")

if __name__ == "__main__":
    main()
