import numpy as np
from stable_baselines3 import PPO
from utils.env_wrapper import make_env


def evaluate(trained_model, robot, task, episodes):
    env = make_env(robot, task)
    model = PPO.load(trained_model)
    successes = 0
    episode_reward = []
    episode_steps = []
    episode_final_distances = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if terminated:
                successes += 1
        episode_reward.append(total_reward)
        episode_steps.append(steps)
        episode_final_distances.append(info["distance"])

    env.close()
    results = {
        "success_rate": round(successes / episodes, 4),
        "avg_reward": round(np.mean(episode_reward), 4),
        "avg_steps": round(np.mean(episode_steps), 4),
        "avg_final_distance": round(np.mean(episode_final_distances), 4)
    }
    return results
