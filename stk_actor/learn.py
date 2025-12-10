# learn.py

import torch
from pathlib import Path
import inspect
from functools import partial

from pystk2_gymnasium import AgentSpec
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from bbrl.agents.gymnasium import make_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .actors import SB3Actor
from .pystk_actor import env_name, get_wrappers, player_name




CHECKPOINT_DIR = Path("/Vrac/TD_proj/checkpoints")
LOG_DIR = Path("/Vrac/TD_proj/logs")


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
    env = Monitor(env)
    env = DummyVecEnv([base_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)
    print("sample obs:", env.reset()[0])
    print("sample action:", env.action_space.sample())

    # check_env(env, warn=True)
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    
    checkpoint_on_event = CheckpointCallback(save_freq=1, 
                                             save_path=CHECKPOINT_DIR, 
                                             save_replay_buffer=True)
    event_callback = EveryNTimesteps(n_steps=5000, callback=checkpoint_on_event)
    checkpoints = sorted(CHECKPOINT_DIR.glob("rl_model_*_steps.zip"))


    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],
            qf=[512, 512, 256]
        ),
        activation_fn=torch.nn.ReLU
    )


    if checkpoints:
        last_checkpoint = checkpoints[-1]
        print("Load checkpoint:", last_checkpoint)

        # model = SAC.load(last_checkpoint, env)
        model = SAC.load(last_checkpoint, env=env, tensorboard_log=str(LOG_DIR))
        model.num_timesteps = int(last_checkpoint.stem.split("_")[2])
        replay_buf_path = last_checkpoint.with_suffix("").as_posix() + "_replay_buffer.pkl"
        if Path(replay_buf_path).exists():
            model.load_replay_buffer(replay_buf_path)
            print("Load replay buffer.")
        else:
            print("replay buffer not found")

    else:
        print("start from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=512,
            train_freq=1,
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
            target_update_interval=1,
            learning_starts=20_000,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(LOG_DIR)
        )

    model.learn(total_timesteps=1000_000,tb_log_name="run_1", progress_bar = True, callback=event_callback)

    policy = model.policy

    sb3_actor = SB3Actor(model)
    print("Model will be saved to:", mod_path / "models/model.zip")
    torch.save(policy.state_dict(), mod_path / "pystk_actor.pth")
    model.save(mod_path / "models/model.zip")
    env.save(mod_path / "models/vecnormalize.pkl")

if __name__ == "__main__":
    main()
