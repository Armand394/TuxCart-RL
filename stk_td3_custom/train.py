import torch
from loss_functions import compute_td3_actor_loss, compute_td3_critic_loss
from utils import setup_optimizer

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Polyak averaging: target = tau*source + (1-tau)*target"""
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

def train_td3_step(
    actor, critic1, critic2,
    target_actor, target_critic1, target_critic2,  # target_actor is ignored (kept for call compatibility)
    replay_buffer,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_delay=2,
    noise_sigma=0.2,
    noise_clip=0.5,
    device="cpu",
    max_action=1.0,   # since your actor ends with Tanh(), this is typically 1.0
    actor_lr=1e-3,
    critic_lr=1e-3,
):
    """
    One TD3 update step:
    - Update both critics every call
    - Update actor + soft-update target critics every `policy_delay` calls
    - Uses *main actor* for target action (no target actor)
    """

    # --- Safety: don’t train until buffer has enough data
    if replay_buffer.size < batch_size:
        return None

    # --- Create optimizers lazily (so main() doesn’t need to)
    if not hasattr(actor, "_optimizer"):
        actor._optimizer = setup_optimizer(actor, lr=actor_lr)
    if not hasattr(critic1, "_optimizer") or not hasattr(critic2, "_optimizer"):
        # You can use one optimizer for both critics, or separate ones.
        # Separate makes it explicit and avoids any accidental coupling.
        critic1._optimizer = setup_optimizer(critic1, lr=critic_lr)
        critic2._optimizer = setup_optimizer(critic2, lr=critic_lr)

    # --- Internal step counter for delayed policy updates
    if not hasattr(train_td3_step, "_it"):
        train_td3_step._it = 0
    train_td3_step._it += 1

    # --- Sample batch (already on correct device per your ReplayBuffer)
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # ============================================================
    # 1) Critic target: y = r + gamma*(1-done)*min(Q1', Q2')
    #    using target critics and *no target actor*:
    #    next_action = actor(next_states) + clipped_noise
    # ============================================================
    with torch.no_grad():
        # Target policy smoothing (TD3 trick)
        next_actions = actor(next_states)

        noise = torch.randn_like(next_actions) * noise_sigma
        noise = noise.clamp(-noise_clip, noise_clip)

        next_actions = (next_actions + noise).clamp(-max_action, max_action)

        target_q1 = target_critic1(next_states, next_actions)
        target_q2 = target_critic2(next_states, next_actions)
        target_q  = torch.min(target_q1, target_q2)  # Clipped Double-Q

    # ============================================================
    # 2) Critic updates: minimize TD error for each critic
    # ============================================================
    q1 = critic1(states, actions)
    q2 = critic2(states, actions)

    critic1_loss = compute_td3_critic_loss(q1, target_q, rewards, dones, gamma)
    critic2_loss = compute_td3_critic_loss(q2, target_q, rewards, dones, gamma)

    critic1._optimizer.zero_grad(set_to_none=True)
    critic1_loss.backward()
    critic1._optimizer.step()

    critic2._optimizer.zero_grad(set_to_none=True)
    critic2_loss.backward()
    critic2._optimizer.step()

    info = {
        "critic1_loss": float(critic1_loss.detach().cpu()),
        "critic2_loss": float(critic2_loss.detach().cpu()),
    }

    # ============================================================
    # 3) Delayed actor update + target critic soft updates
    # ============================================================
    if train_td3_step._it % policy_delay == 0:
        # Actor aims to maximize Q1(s, pi(s)) => minimize -Q1(...)
        pi_actions = actor(states)
        q1_pi = critic1(states, pi_actions)
        actor_loss = compute_td3_actor_loss(q1_pi)

        actor._optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor._optimizer.step()

        # Soft update target critics (standard TD3 does this on delayed steps)
        soft_update(target_critic1, critic1, tau)
        soft_update(target_critic2, critic2, tau)

        info["actor_loss"] = float(actor_loss.detach().cpu())

    return info
