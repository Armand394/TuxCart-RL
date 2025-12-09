
from .actors import Actor, SB3Actor
from .wrappers import  FlattenActionSpace, FlattenDictWrapper
from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

env_name = "supertuxkart/flattened-v0"
player_name = "MyTD3Agent"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: FlattenDictWrapper(env),
        lambda env: FlattenActionSpace(env),
        lambda env: Monitor(env),
        lambda env: DummyVecEnv([lambda: env]),
        lambda env: VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    ]


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    path = "stk_actor/models/model"
    model = SAC.load(path)
    policy = model.policy
    actor = SB3Actor(policy, deterministic=False)
    argmax_actor = SB3Actor(policy, deterministic=True)
    return Agents(actor, argmax_actor)
