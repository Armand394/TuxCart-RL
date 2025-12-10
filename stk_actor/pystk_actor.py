
from .actors import Actor, SB3Actor, SamplingActor
from .wrappers import  FlattenActionSpace, FlattenDictWrapper, FlattenFilterWrapper
from typing import List, Callable
import gymnasium as gym
from bbrl.agents import Agents, Agent
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from pystk2_gymnasium import PolarObservations
# env_name = "supertuxkart/flattened-v0"
# env_name="supertuxkart/flattened_continuous_actions-v0"
env_name = "supertuxkart/full-v0"
player_name = "MySACAgent"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        # lambda env: FlattenDictWrapper(env),
        # lambda env: FlattenActionSpace(env),
        lambda env: PolarObservations(env),
        lambda env: FlattenFilterWrapper(env),
        lambda env: FlattenActionSpace(env),
    ]


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    path = "stk_actor/models/model_fix.zip"
    model = SAC.load(path)
    policy = model.policy
    actor = SB3Actor(policy, deterministic=False)
    argmax_actor = SB3Actor(policy, deterministic=True)
    print("sample action space:", action_space.sample())
    print("observation space:", observation_space.sample())
    return Agents(actor, argmax_actor)

# def get_actor(
#     state: dict | None,
#     observation_space: gym.spaces.Space,
#     action_space: gym.spaces.Space,
# ) -> Agent:
#     """Creates a new actor (BBRL agent) that write into `action`

#     :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
#     :param observation_space: The environment observation space (with wrappers)
#     :param action_space: The environment action space (with wrappers)
#     :return: a BBRL agent
#     """
#     # actor = Actor(observation_space, action_space)


#     return SamplingActor(action_space)
