from stable_baselines3.common.callbacks import BaseCallback
import torch
import numpy as np

class FeatureImportanceBarCallback(BaseCallback):
    def __init__(self, env, eval_freq=5000, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True

        try:
            mean_obs = self.env.get_attr("obs_rms")[0].mean
        except:
            mean_obs = self.env.envs[0].env.observation_space.sample()

        obs = torch.tensor(mean_obs, dtype=torch.float32).unsqueeze(0)
        obs.requires_grad_(True)

        action, _ = self.model.policy(obs, deterministic=True)

        action.sum().backward()

        grad = obs.grad.abs().detach().cpu().numpy()[0] 

        self.logger.record("feature_importance/bar", grad)

        return True