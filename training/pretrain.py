from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.mixed_arm_env import MixedArmEnv
from utils.experiment_config import CONFIG


def pretrain(robot_ratio, task_ratio):
    cfg = CONFIG
    env = MixedArmEnv(
        robots=["panda", "kuka"],
        robot_ratio=robot_ratio,
        tasks=["reach", "push"],
        task_ratio=task_ratio,
    )
    env = DummyVecEnv([lambda: env])
    model = PPO(
        policy=cfg["policy"],
        env=env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        policy_kwargs=cfg["policy_kwargs"],
        tensorboard_log=cfg["tensorboard_log"],
        verbose=cfg["verbose"],
        seed=cfg["seed"],
        device=cfg["device"],
    )
    model.learn(
        total_timesteps=cfg["pt_total_timesteps"],
        progress_bar=cfg["progress_bar"]
    )
    savepath = f"../models/pt_r{robot_ratio[0]}{robot_ratio[1]}_t{task_ratio[0]}{task_ratio[1]}"
    model.save(savepath)
    env.close()
    return savepath
