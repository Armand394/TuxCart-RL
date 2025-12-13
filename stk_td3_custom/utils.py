import torch
import numpy as np
from torch import nn, optim
from itertools import chain
from typing import Union
import gymnasium as gym
from cartPole_continuous import ContinuousCartPoleEnv

def setup_optimizer(*agents: Union[nn.Module, nn.Parameter], lr=1e-3, eps=5e-5):
    """
    Creates a single Adam optimizer for any number of torch modules or parameters.

    Example:
        actor_opt = setup_optimizer(actor)
        critic_opt = setup_optimizer(critic1, critic2)
        full_opt   = setup_optimizer(actor, critic1, critic2)
    """

    # Collect all parameters from modules or parameters
    all_params = []

    for agent in agents:
        if isinstance(agent, nn.Module):
            all_params.append(agent.parameters())
        elif isinstance(agent, nn.Parameter):
            all_params.append([agent])
        else:
            raise ValueError(
                f"Unsupported type {type(agent)}. "
                "Expected nn.Module or nn.Parameter."
            )

    # Flatten into a single iterator of parameters
    all_params = chain(*all_params)

    optimizer = optim.Adam(all_params, lr=lr, eps=eps)
    return optimizer

def make_env(max_episode_steps):
    # env = ContinuousCartPoleEnv()
    env = gym.make("LunarLanderContinuous-v3")
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

@torch.no_grad()
def evaluate_policy(actor, max_episode_steps, n_runs=3, device=None):
    """
    Runs 1 episode in each of n_envs environments *in parallel* (manual parallel stepping),
    using deterministic actor actions (no exploration noise), and returns mean episode return.
    """

    # Make multiple independent envs
    envs = [make_env(max_episode_steps) for _ in range(n_runs)]
    obs = []
    for e in envs:
        s, _ = e.reset()
        obs.append(s)

    obs = np.stack(obs, axis=0)  # shape: (n_envs, obs_dim)
    done = np.zeros(n_runs, dtype=bool)
    ep_ret = np.zeros(n_runs, dtype=np.float32)

    # For clipping actions (same across envs)
    act_low = envs[0].action_space.low
    act_high = envs[0].action_space.high

    # Decide which device to run actor on
    actor_device = next(actor.parameters()).device if device is None else torch.device(device)

    # Ensure eval mode (even if you don't use dropout/BN, it's clean)
    was_training = actor.training
    actor.eval()

    while not done.all():
        # Build batch tensor
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=actor_device)

        # Deterministic actions
        actions = actor(obs_tensor).cpu().numpy()
        actions = np.clip(actions, act_low, act_high)

        # Step each env only if not done
        for i, env in enumerate(envs):
            if done[i]:
                continue
            ns, r, terminated, truncated, _ = env.step(actions[i])
            ep_ret[i] += r
            done[i] = terminated or truncated
            obs[i] = ns

    # Close envs
    for env in envs:
        env.close()

    # Restore actor mode
    actor.train(was_training)

    return float(ep_ret.mean())


def linear_sigma(t, start_t, decay_steps, sigma_start, sigma_end):
    # decay begins after warmup; clamps to sigma_end
    x = (t - start_t) / max(1, decay_steps)
    x = float(np.clip(x, 0.0, 1.0))
    return sigma_start + x * (sigma_end - sigma_start)