import torch



def compute_td3_critic_loss(q, target_q, rewards, dones, gamma):
    """
    q:         Q(s,a) predicted by critic        shape: (batch, 1)
    target_q:  target Q(s',a')                  shape: (batch, 1)
    rewards:   r                                shape: (batch, 1)
    dones:     done flags (1 = terminal)        shape: (batch, 1)
    """
    # TD target: y = r + gamma * (1-done) * target_q
    y = rewards + gamma * (1 - dones.float()) * target_q

    td_error = q - y.detach()
    loss = (td_error ** 2).mean()
    return loss



def compute_td3_actor_loss(q_pi):
    """
    q_pi: Q1(s, pi(s)) — critic evaluated on actor-chosen action
          shape: (batch, 1)
    """
    return -q_pi.mean()

