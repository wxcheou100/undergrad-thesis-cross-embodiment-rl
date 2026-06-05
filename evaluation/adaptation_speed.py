import os
import re
from evaluation.evaluate import evaluate


def adaptation_speed(log_dir, robot, task, threshold, episodes):
    files = os.listdir(log_dir)
    checkpoints = []
    for file in files:
        if file.startswith("checkpoint") and file.endswith(".zip"):
            timesteps = int(re.findall(r"\d+", file)[0])
            checkpoints.append((timesteps, file))
    checkpoints.sort()
    threshold_timesteps = None
    for _, (timesteps, file) in enumerate(checkpoints):
        cp_model = os.path.join(log_dir, file)
        cp_results = evaluate(cp_model, robot, task, episodes=episodes)
        cp_success_rate = cp_results["success_rate"]
        if cp_success_rate >= threshold:
            threshold_timesteps = timesteps
            break
    return {
        "adapt_timesteps_to_threshold": threshold_timesteps
    }
