import os
import time
import gymnasium as gym
import numpy as np
import torch
from collections import deque
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from Agents import Actor, Critic, GaussianNoise
from replayBuffer import ReplayBuffer
from train import train_td3_step
from utils import evaluate_policy, make_env, linear_sigma
from collections import deque
from cartPole_continuous import ContinuousCartPoleEnv

def main(cfg):

    # Reproducibility
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Env
    # ----------------------------
    # env = ContinuousCartPoleEnv()
    env = gym.make("LunarLanderContinuous-v3")    
    env = gym.wrappers.TimeLimit(env, max_episode_steps=cfg["max_episode_steps"])

    env.action_space.seed(cfg["seed"])

    obs_space = env.observation_space
    act_space = env.action_space

    print(f'Spaces obs={obs_space}, act={act_space}')

    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]

    max_action = float(act_space.high[0])  # should be 1.0 in your wrapper

    # ----------------------------
    # TensorBoard
    # ----------------------------
    log_path = os.path.join(cfg["log_dir"], cfg["run_name"])
    writer = SummaryWriter(log_dir=log_path)

    # log config once
    writer.add_text("config", str(cfg))

    # ----------------------------
    # Replay buffer (on device)
    # ----------------------------
    replay_buffer = ReplayBuffer(
        state_dim=obs_dim,
        action_dim=act_dim,
        max_capacity=cfg["buffer_capacity"],
        device=str(device),
    )

    # ----------------------------
    # Networks
    # IMPORTANT: pass spaces (not dims)
    # ----------------------------
    actor = Actor(obs_space, act_space, hidden1=cfg["hidden1"], hidden2=cfg["hidden2"]).to(device)

    critic1 = Critic(obs_space, act_space, hidden1=cfg["hidden1"], hidden2=cfg["hidden2"]).to(device)
    critic2 = Critic(obs_space, act_space, hidden1=cfg["hidden1"], hidden2=cfg["hidden2"]).to(device)

    # Target critics (deep copy)
    # You used critic.copy(deep=True) in main, but your Critic class doesn't define copy().
    # So use state_dict cloning (safe + standard).
    target_critic1 = Critic(obs_space, act_space, hidden1=cfg["hidden1"], hidden2=cfg["hidden2"]).to(device)
    target_critic2 = Critic(obs_space, act_space, hidden1=cfg["hidden1"], hidden2=cfg["hidden2"]).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    # Exploration noise module
    exploration_noise = GaussianNoise(sigma_start=cfg["exploration_sigma_start"])

    # ----------------------------
    # Training loop
    # ----------------------------
    s, _ = env.reset(seed=cfg["seed"])

    eval_window = cfg.get("eval_ma_window", 10)   # moving average window over eval points
    eval_queue = deque(maxlen=eval_window)
    eval_ma = float("nan")

    # Training episode stats
    ep_ret, ep_len = 0.0, 0

    # tqdm: reduce terminal load on SSH
    pbar = tqdm(
        range(cfg["total_timesteps"]),
        desc="TD3 Training",
        dynamic_ncols=False,     # safer over SSH than True
        mininterval=0.5,         # don't redraw too aggressively
    )


    for t in pbar:
        # action selection
        if t < cfg["start_timesteps"]:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a = actor(s_t).cpu().numpy()[0]


            # update exploration sigma (slow linear decay)
            exploration_noise.sigma = linear_sigma(
                t,
                start_t=cfg["start_timesteps"],
                decay_steps=cfg["exploration_decay_steps"],
                sigma_start=cfg["exploration_sigma_start"],
                sigma_end=cfg["exploration_sigma_end"],
            )

            # add exploration noise for behavior policy
            a = exploration_noise(torch.as_tensor(a)).numpy()
            a = np.clip(a, env.action_space.low, env.action_space.high)

        # ----------------------------
        # Useful TB logs (low overhead)
        # ----------------------------
        if (t + 1) % cfg["log_freq"] == 0 and t >= cfg["start_timesteps"]:
            # executed action stats
            a0 = float(a[0])
            writer.add_scalar("train/action_mean", a0, t)
            writer.add_scalar("train/action_abs_mean", abs(a0), t)
            writer.add_scalar("train/action_frac_saturated", float(abs(a0) > 0.95 * max_action), t)

            # exploration sigma
            writer.add_scalar("train/exploration_sigma", float(exploration_noise.sigma), t)

            # cheap critic sanity check on current state (on-policy)
            with torch.no_grad():
                s_t = torch.as_tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
                a_pi = actor(s_t)
                q1_pi = critic1(s_t, a_pi)
                writer.add_scalar("debug/q1_pi_mean", float(q1_pi.item()), t)

        # env step
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        # track train episode stats
        ep_ret += float(r)
        ep_len += 1

        # store transition (noisy action is correct)
        replay_buffer.add(s, a, r, s2, float(done))
        s = s2 if not done else env.reset()[0]

        if done:
            # log train episode stats
            writer.add_scalar("train/episode_return", ep_ret, t)
            writer.add_scalar("train/episode_length", ep_len, t)
            ep_ret, ep_len = 0.0, 0

            s, _ = env.reset()

        # train
        if t >= cfg["start_timesteps"]:
            info = train_td3_step(
                actor, critic1, critic2,
                None, target_critic1, target_critic2,   # no target actor
                replay_buffer,
                batch_size=cfg["batch_size"],
                gamma=cfg["gamma"],
                tau=cfg["tau"],
                policy_delay=cfg["policy_delay"],
                noise_sigma=cfg["target_noise_sigma"],
                noise_clip=cfg["target_noise_clip"],
                device=str(device),
                max_action=max_action,
                actor_lr=cfg["actor_lr"],
                critic_lr=cfg["critic_lr"],
            )

            # TensorBoard logging (losses)
            if info is not None and (t + 1) % cfg["tb_freq"] == 0:
                writer.add_scalar("loss/critic1", info["critic1_loss"], t)
                writer.add_scalar("loss/critic2", info["critic2_loss"], t)
                if "actor_loss" in info:
                    writer.add_scalar("loss/actor", info["actor_loss"], t)

        # log buffer growth
        if (t + 1) % 1000 == 0:
            writer.add_scalar("buffer/size", replay_buffer.size, t)

        # periodic evaluation
        if (t + 1) % cfg["eval_freq"] == 0:
            pbar.write(f"Starting eval at t={t+1}")
            mean_ret = evaluate_policy(actor, max_episode_steps=cfg["max_episode_steps"], device=device, n_runs=cfg["eval_runs"])
            pbar.write(f"Finished eval at t={t+1}")

            writer.add_scalar("eval/mean_return", mean_ret, t)

            eval_queue.append(mean_ret)
            eval_ma = float(np.mean(eval_queue))

            pbar.write(f"[Eval] t={t+1} mean_return={mean_ret:.2f}")

        if (t + 1) % 1000 == 0:
            pbar.set_postfix(
                t=t+1,
                eval_ma=f"{eval_ma:.2f}" if not np.isnan(eval_ma) else "nan",
                buffer=replay_buffer.size
            )

    env.close()
    writer.close()


if __name__ == "__main__":
    # ----------------------------
    # Config (easy tuning)
    # ----------------------------
    cfg = {
        "seed": 0,
        "total_timesteps": 300_000,
        "start_timesteps": 10_000,
        "max_episode_steps": 1_000,

        "buffer_capacity": 200_000,
        "batch_size": 256,

        "gamma": 0.98,
        "tau": 0.005,
        "policy_delay": 2,

        # TD3 target smoothing noise (training target only)
        "target_noise_sigma": 0.1,
        "target_noise_clip": 0.3,

        "actor_lr": 1e-3,
        "critic_lr": 1e-3,

        "hidden1": 400,
        "hidden2": 300,

        "eval_freq": 5_000,
        "eval_runs": 3,

        "log_dir": "runs",
        "run_name": f"td3_lunarlander_{int(time.time())}",
        "eval_ma_window": 10,

        # exploration noise (env action selection)
        "exploration_sigma_start": 0.15,
        "exploration_sigma_end": 0.03,
        "exploration_decay_steps": 300_000,  # slow decay over most of training

        # logging frequency
        "log_freq": 1000,                # how often to log action stats
        "tb_freq": 500,                  # how often to log losses to tensorboard
    }

    main(cfg)