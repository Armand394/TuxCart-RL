import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.env_checker import check_env
from pystk2_gymnasium import AgentSpec
from wrappers import STKContinuousWrapper
from tqdm import tqdm


# STK gymnasium uses one process
if __name__ == "__main__":
    # Base environment
    base_env = gym.make(
        "supertuxkart/flattened-v0",
        agent=AgentSpec(use_ai=False),
        difficulty=0,
        render_mode="human"
    )

    # Wrapped environment
    env = STKContinuousWrapper(base_env)

    print("Obs space:", env.observation_space)
    print("Action space:", env.action_space)

    check_env(env, warn=True)

    # ======================
    # Train TD3
    # ======================
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        train_freq=(2, "episode"),
        tensorboard_log="./logs/tuxCart-td3-continuous-tb/"
    )

    model.learn(total_timesteps=200_000, tb_log_name="run_1", progress_bar = True)

    # Save
    model.save("td3_stk_continuous")


    env.close()