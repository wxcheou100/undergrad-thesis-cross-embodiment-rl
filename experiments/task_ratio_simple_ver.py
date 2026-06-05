import os
import csv
from training.pretrain import pretrain
from training.finetune import finetune
from evaluation.evaluate import evaluate
from evaluation.adaptation_speed import adaptation_speed

task_ratios = [
    # [100, 0],
    # [75, 25],
    [50, 50],
    # [25, 75],
    # [0, 100]
]
robot_ratio = [50, 50]
tasks = ["reach", "push", "moving"]
robots = ["panda", "kuka"]
results = []

for task_ratio in task_ratios:
    # Pretraining
    pt_model = pretrain(robot_ratio, task_ratio)
    for task in tasks:
        for robot in robots:
            zero_results = evaluate(pt_model, robot, task, episodes=20)
            zero_results = {f"zero_{k}": v for k, v in zero_results.items()}
            # Finetuning
            log_dir, ft_model = finetune(pt_model, robot, task)
            # Evaluation
            ft_results = evaluate(ft_model, robot, task, episodes=20)
            ft_results = {f"ft_{k}": v for k, v in ft_results.items()}
            improvement = ft_results["ft_success_rate"] - zero_results["zero_success_rate"]
            adapt_results = adaptation_speed(log_dir, robot, task, threshold=0.8, episodes=10)
            results.append({
                "robot_ratio": robot_ratio,
                "task_ratio": task_ratio,
                "robot": robot,
                "task": task,
                **zero_results,
                **ft_results,
                "improvement": improvement,
                **adapt_results
            })

os.makedirs("../results", exist_ok=True)
with open("../results/task_ratio.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("Experiment complete.")
