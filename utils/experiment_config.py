CONFIG = {
    "robots": ["panda", "kuka", "xarm6"],
    "tasks": ["reach", "push", "moving"],
    # during PPO initialization
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": dict(net_arch=[256, 256]),
    "tensorboard_log": None,
    "verbose": 0,
    "seed": 42,
    "device": "auto",
    # during PPO learning
    "pt_total_timesteps": 3_000_000,    # pretraining
    "ft_total_timesteps": 300_000,      # finetuning
    "save_freq": 10_000,
    "progress_bar": False,
    # during evaluation
    "eval_episodes": 100,   # evaluate
    "adapt_episodes": 50,   # adaptation speed
    "threshold": 0.6        # adaptation success threshold
}
