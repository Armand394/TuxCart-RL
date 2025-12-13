# learn.py

import torch
from pathlib import Path
import inspect
from functools import partial
import os
from pystk2_gymnasium import AgentSpec
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from bbrl.agents.gymnasium import make_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .actors import SB3Actor
from .pystk_actor import env_name, get_wrappers, player_name,get_wrappers_train
from .callback import FeatureImportanceBarCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
#import gestion des arguments en appelant le script
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("from_scratch", type=bool, default=True, help="True to train from scratch, False to load last checkpoint")
args = parser.parse_args()

CHECKPOINT_DIR = Path("/Vrac/TD_proj/checkpoints")
LOG_DIR = Path("/Vrac/TD_proj/logs")

def load_vecnormalize(env, save_path):
    if save_path.exists():
        print("Load VecNormalize:", save_path)
        env = VecNormalize.load(save_path, env)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env


def main():              
    base_env = partial(
        make_env,
        env_name,
        wrappers=get_wrappers_train(),
        render_mode=None,
        difficulty =0,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    check_env(base_env(), warn=True)
    env = DummyVecEnv([
        lambda: Monitor(base_env())
    ])



    vecnorm_path = Path("models/vecnormalize_best_param.pkl")
    env = load_vecnormalize(env, vecnorm_path)
    # env = VecNormalize(env, norm_obs=True, norm_reward=False)
    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)
    
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    
    checkpoint_on_event = CheckpointCallback(save_freq=1, 
                                             save_path=CHECKPOINT_DIR, 
                                             save_replay_buffer=True)
    event_callback = EveryNTimesteps(n_steps=5000, callback=checkpoint_on_event)
    feature_cb = FeatureImportanceBarCallback(env, eval_freq=5000)

    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256],
            qf=[256, 256]
        ),
        activation_fn=torch.nn.ReLU
    )

    checkpoints = sorted(CHECKPOINT_DIR.glob("rl_model_*_steps.zip"))

    if checkpoints and not args.from_scratch:
        last_checkpoint = checkpoints[-1]
        print("Load checkpoint:", last_checkpoint)

        # model = SAC.load(last_checkpoint, env)
        model = SAC.load(last_checkpoint, env=env, tensorboard_log=str(LOG_DIR))
        steps = last_checkpoint.stem.split("_")[2]
        model.num_timesteps = int(steps)
        print("num steps =", model.num_timesteps)

        if os.path.exists(last_checkpoint.parent / f"rl_model_replay_buffer_{steps}_steps.pkl"):
            replay_buf_path = last_checkpoint.parent / f"rl_model_replay_buffer_{steps}_steps.pkl"
            print("replay buffer:", replay_buf_path)
            model.load_replay_buffer(replay_buf_path)

        else:
            print("rreplay buffer not found")


    else:
        print("start from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=2.1573e-05,
            buffer_size=1_000_000,
            batch_size=256,
            train_freq=4,
            gradient_steps=4,
            gamma=0.99499,
            tau=0.005,
            target_update_interval=1,
            learning_starts=70_000,
            ent_coef="auto",
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(LOG_DIR)
        )

    model.learn(total_timesteps=500_000,tb_log_name="new_accel", progress_bar = True, callback=[event_callback])

    policy = model.policy

    sb3_actor = SB3Actor(model)
    suffixe = "_new_accel"
    print("Model will be saved to:", mod_path / f"models/model{suffixe}.zip")
    torch.save(policy.state_dict(), mod_path / f"pystk_actor{suffixe}.pth")
    model.save(mod_path / f"models/model{suffixe}.zip")
    env.save(mod_path / f"models/vecnormalize{suffixe}.pkl")



if __name__ == "__main__":
    main()
