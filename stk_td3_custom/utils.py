import torch
from torch import nn, optim
from itertools import chain
from typing import Union


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