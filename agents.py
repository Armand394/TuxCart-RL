from torch import nn
import enum

class ActionSpace(enum):
    DISCRETE = 0
    CONTINUOUS = 1
    HYBRID = 2



class Actor(nn.Module):
    def __init__(self, state_dim : int, action_dim: int, hidden_layers : list[int], action_space: ActionSpace, hybrid_params : list[int] = None):
        
        if len(hidden_layers) != 2:
            raise ValueError("The hidden layer list for the actor network should be of size 2.")
        
        # Hidden Layers
        hidden1, hidden2 = hidden_layers[0], hidden_layers[1]
        

        # Full Continuous action space
        if action_space == ActionSpace.CONTINUOUS:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden1),
                            nn.Tanh(),
                            nn.Linear(hidden1, hidden2),
                            nn.Tanh(),
                            nn.Linear(hidden2, action_dim),
                            nn.Tanh()
                        )
        
        # Full Discrete action space
        elif action_space == ActionSpace.DISCRETE:
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, hidden1),
                            nn.Tanh(),
                            nn.Linear(hidden1, hidden2),
                            nn.Tanh(),
                            nn.Linear(hidden2, action_dim),
                            nn.Softmax(dim=-1)
                        )
            
        # Hybdrid type action space
        else:

            if hybrid_params:

                self.actor = nn.Sequential(
                                nn.Linear(state_dim, hidden1),
                                nn.Tanh(),
                                nn.Linear(hidden1, hidden2),
                                nn.Tanh(),
                                nn.Linear(hidden2, action_dim),
                                nn.Softmax(dim=-1)
                            )
