
from .actors import Actor, SB3Actor, SamplingActor, ArgmaxActor
from .wrappers import OnlySteerAction,OnlySteerActionEval, FlattenActionSpace, FlattenActionSpaceEval,FilterWrapper, FilterWrapperEval,ObsNormalizeWrapper
from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from pystk2_gymnasium import PolarObservations
import torch
import numpy as np
# from path import Path
# env_name = "supertuxkart/flattened-v0"
# env_name="supertuxkart/flattened_continuous_actions-v0"
env_name = "supertuxkart/full-v0"
player_name = "MySACAgent"

def get_wrappers_train() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        # lambda env: FlattenDictWrapper(env),
        # lambda env: FlattenActionSpace(env),
        lambda env: PolarObservations(env),
        lambda env: FlattenActionSpace(env),
        # lambda env: OnlySteerAction(env),
        lambda env: FilterWrapper(env),
    ]

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    suffixe ='_os'
    vecnorm_path= Path(f'stk_actor/models/vecnormalize{suffixe}.pkl')
    return [
        # Example of a custom wrapper
        # lambda env: FlattenDictWrapper(env),
        # lambda env: FlattenActionSpace(env),
        lambda env: PolarObservations(env),
        lambda env: FlattenActionSpaceEval(env),
        # lambda env: OnlySteerActionEval(env),
        lambda env: FilterWrapperEval(env),
        lambda env: ObsNormalizeWrapper(env, vecnorm_path),
    ]



# def get_actor(
#     state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
# ) -> Agent:
#     path = "stk_actor/models/model.zip"
#     model = SAC.load(path)
#     policy = model.policy
#     actor = SB3Actor(policy, deterministic=False)
#     argmax_actor = SB3Actor(policy, deterministic=True)
#     print("sample action space:", action_space.sample())
#     print("observation space:", observation_space.sample())
#     return Agents(actor, argmax_actor)


def reconstruct_sac_policy(observation_space, action_space, state_dict):
    dummy_env = DummySTKEnv(observation_space, action_space)

    model = SAC(
        "MlpPolicy",
        dummy_env,
        policy_kwargs=dict(
            net_arch=dict(pi=[256,256], qf=[256,256]),
            # net_arch=dict(pi=[512,512,256], qf=[512,512,256]),
            activation_fn=torch.nn.ReLU
        ),
        verbose=0,
    )

    model.policy.load_state_dict(state_dict)
    return model.policy



class DummySTKEnv(gym.Env):
    #besoin juste pour reconstruire la policy on a besoin d'un env
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {},
        )
    
def get_actor(state, observation_space, action_space):
    if state is None:
        return SamplingActor(action_space)

    policy = reconstruct_sac_policy(observation_space, action_space, state)

    actor = SB3Actor(policy, deterministic=True)
    argmax_actor = SB3Actor(policy, deterministic=True)

    return Agents(actor, argmax_actor)