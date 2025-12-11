# optuna_sac.py
import optuna
from functools import partial
from pathlib import Path
import yaml
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from bbrl.agents.gymnasium import make_env
from pystk2_gymnasium import AgentSpec
from stk_actor.pystk_actor import env_name, get_wrappers_train, player_name
import matplotlib.pyplot as plt

CHECKPOINT_DIR = Path("/Vrac/TD_proj/checkpoints")
LOG_DIR = Path("/Vrac/TD_proj/logs")
PARAMS_DIR = Path("/Vrac/TD_proj/params")
FIGURES_DIR = Path("/Vrac/TD_proj/figures")

def make_sac_env():
    base_env = partial(
        make_env,
        env_name,
        wrappers=get_wrappers_train(),
        render_mode=None,
        difficulty=0,
        agent=AgentSpec(use_ai=False, name=player_name),
    )
    env = DummyVecEnv([lambda: Monitor(base_env())])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    return env


def objective(trial: optuna.Trial):

    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],
            qf=[512, 512, 256]
        ),
        activation_fn=torch.nn.ReLU
    )

    env = make_sac_env()
    model = SAC("MlpPolicy",
                env,
                learning_rate=trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                buffer_size=1_000_000,
                batch_size=512,
                train_freq=1,
                gradient_steps=1,
                gamma=trial.suggest_uniform("gamma", 0.95, 0.9999),
                tau=0.005,
                target_update_interval=1,
                learning_starts=20_000,
                ent_coef=trial.suggest_loguniform("ent_coef", 0.0001, 1.0),
                policy_kwargs=policy_kwargs,
                verbose=0, tensorboard_log=str(LOG_DIR))
    

    model.learn(total_timesteps=100_000)

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)
    
    env.close()
    print("-" * 40)
    print("parameters:", trial.params)
    print(f"Trial {trial.number}: Mean Reward = {mean_reward}")
    return mean_reward




if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)  
    study_analyse = study.trials_dataframe(attrs=('params', 'value')) 

    best_params = study.best_params
    print("Meilleurs hyperparamètres:", best_params)


    with open(PARAMS_DIR, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)

    plt.figure(figsize=(10, 6))
    optuna.visualization.plot_param_importances(study).show()
    plt.save(FIGURES_DIR / "optuna_param_importance.png")

    plt.figure(figsize=(10, 6))
    optuna.visualization.plot_optimization_history(study).show()
    plt.save(FIGURES_DIR / "optuna_optimization_history.png")

    lr = study_analyse['params_learning_rate']
    gamma = study_analyse['params_gamma']
    ent_coef = study_analyse['params_ent_coef']
    rewards = study_analyse['value']

    fig,axi =plt.subplots(1,3,figsize=(18,5))
    sc1 = axi[0].scatter(lr, gamma, c=rewards, cmap="RdYlGn_r")
    plt.colorbar(sc1, ax=axi[0], label='Mean Reward')
    axi[0].set_xlabel('Learning Rate')
    axi[0].set_ylabel('Gamma')
    axi[0].set_title('Learning Rate vs Gamma')
    sc2 = axi[1].scatter(lr, ent_coef, c=rewards, cmap="RdYlGn_r")
    plt.colorbar(sc2, ax=axi[1], label='Mean Reward')
    axi[1].set_xlabel('Learning Rate')
    axi[1].set_ylabel('Entropy Coefficient')
    axi[1].set_title('Learning Rate vs Entropy Coefficient')
    sc3 = axi[2].scatter(gamma, ent_coef, c=rewards, cmap="RdYlGn_r")
    plt.colorbar(sc3, ax=axi[2], label='Mean Reward')
    axi[2].set_xlabel('Gamma')
    axi[2].set_ylabel('Entropy Coefficient')
    axi[2].set_title('Gamma vs Entropy Coefficient')
    plt.savefig(FIGURES_DIR / "optuna_hyperparameter_scatter.png")

