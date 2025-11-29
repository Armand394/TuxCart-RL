import gymnasium as gym
import torch
from stk_td3_custom.Agents import Actor, Critic, GaussianNoise
from stk_td3_custom.wrappers import CartPoleContinuousWrapper
from stk_td3_custom.loss_functions import compute_td3_actor_loss, compute_td3_critic_loss
from stk_td3_custom.replayBuffer import ReplayBuffer
from pystk2_gymnasium import AgentSpec
import numpy as np

def main():

    # Base environment
    env = gym.make(
        "CartPole-v1",
    )

    env = CartPoleContinuousWrapper(env)

    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)

    size_obs, size_action = len(env.observation_space), len(env.action_space)

    buffer_capacity = 1000000   # 1M is standard for TD3
    batch_size = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    replay_buffer = ReplayBuffer(
        state_dim=size_obs,
        action_dim=size_action,
        max_capacity=buffer_capacity,
        device=device
    )

    # Critic 1
    critic1 = Critic(observation_space=size_obs, action_space=size_action, hidden1=400, hidden2=300)
    target_critic1 = critic1.copy(deep=True)

    # Critic 2
    critic2 = Critic(observation_space=size_obs, action_space=size_action, hidden1=400, hidden2=300)
    target_critic2 = critic2.copy(deep=True)
    
    # Actor
    actor = Actor(observation_space=size_obs, action_space=size_action, hidden1=400, hidden2=300)

    # Exploration noise
    exploration_actor = GaussianNoise(sigma_start=0.1)


    total_timesteps = 200_000
    start_timesteps = 5000        # warm-up before training
    state, _ = env.reset()

    for t in range(total_timesteps):

        # Select action
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)

        if t < start_timesteps:
            # Pure random actions at the beginning
            action = env.action_space.sample()
        else:
            # Actor action (deterministic)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]

            # Add exploration noise
            action = exploration_actor(torch.tensor(action)).numpy()

            # Clip to env bounds
            action = np.clip(action, env.action_space.low, env.action_space.high)

        # -----------------------------
        # Step environment
        # -----------------------------
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # -----------------------------
        # Store transition in replay buffer
        # -----------------------------
        replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            float(done)
        )

        # Move to next state
        state = next_state

        if done:
            state, _ = env.reset()

        # -----------------------------
        # Learn AFTER warm-up
        # -----------------------------
        if t >= start_timesteps:
            train_td3_step(
                actor, critic1, critic2,
                target_actor, target_critic1, target_critic2,
                replay_buffer,
                batch_size=batch_size,
                gamma=0.99,
                tau=0.005,
                policy_delay=2,
                noise_sigma=0.2,
                noise_clip=0.5,
                device=device
            )