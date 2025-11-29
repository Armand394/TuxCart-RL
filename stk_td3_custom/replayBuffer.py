import torch

class ReplayBuffer:
    """
    Replay Buffer for Off-Policy RL (TD3, DDPG, SAC).
    
    Stores:
    - states
    - actions
    - rewards
    - next_states
    - terminal flags (done transitions)

    Allows random sampling of batches for training.
    """

    def __init__(self, state_dim, action_dim, max_capacity, device):
        self.max_capacity = max_capacity       # Max number of stored transitions
        self.device = device

        self.ptr = 0                           # Current index for writing new data
        self.size = 0                          # Number of valid items in buffer (<= max_capacity)

        # Storage buffers
        self.states      = torch.zeros((max_capacity, state_dim),  dtype=torch.float32, device=device)
        self.actions     = torch.zeros((max_capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards     = torch.zeros((max_capacity, 1),          dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_capacity, state_dim),  dtype=torch.float32, device=device)
        self.dones       = torch.zeros((max_capacity, 1),          dtype=torch.bool,    device=device)

    def add(self, state, action, reward, next_state, done):
        """
        Adds a single transition: (s, a, r, s_next, done)
        """

        self.states[self.ptr]      = torch.as_tensor(state,      device=self.device)
        self.actions[self.ptr]     = torch.as_tensor(action,     device=self.device)
        self.rewards[self.ptr]     = torch.as_tensor([reward],   device=self.device)
        self.next_states[self.ptr] = torch.as_tensor(next_state, device=self.device)
        self.dones[self.ptr]       = torch.as_tensor([done],     device=self.device)

        # Move pointer forward and wrap at capacity
        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size):
        """
        Returns a random batch of transitions
        for training the actor-critic networks.
        """
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx]
        )
