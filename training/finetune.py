import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env_wrapper import make_env
from utils.experiment_config import CONFIG


def finetune(pretrained_model, robot, task):
    base = os.path.basename(pretrained_model)
    base = base.replace(".zip", "")
    base = base.replace("pt_", "")
    log_dir = f"../logs/ft_{base}_{robot}_{task}/"
    os.makedirs(log_dir, exist_ok=True)
    savepath = f"../models/ft_{base}_{robot}_{task}"
    cfg = CONFIG
    env = make_env(robot, task)
    model = PPO.load(pretrained_model, env=env, device=cfg["device"])
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["save_freq"],
        save_path=log_dir,
        name_prefix="checkpoint"
    )
    model.learn(
        total_timesteps=cfg["ft_total_timesteps"],
        progress_bar=cfg["progress_bar"],
        callback=checkpoint_callback,
    )
    model.save(savepath)
    env.close()
    return log_dir, savepath
