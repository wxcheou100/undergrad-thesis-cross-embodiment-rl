import os
import csv
from training.pretrain import pretrain
from training.finetune import finetune
from evaluation.evaluate import evaluate
from evaluation.adaptation_speed import adaptation_speed

robot_ratios = [
    # [100, 0],
    # [75, 25],
    [50, 50],
    # [25, 75],
    # [0, 100]
]
task_ratio = [50, 50]
robots = ["panda", "kuka", "xarm6"]
tasks = ["reach", "push"]
results = []

for robot_ratio in robot_ratios:
    # Pretraining
    pt_model = pretrain(robot_ratio, task_ratio)
    for robot in robots:
        for task in tasks:
            zero_results = evaluate(pt_model, robot, task, episodes=10)
            zero_results = {f"zero_{k}": v for k, v in zero_results.items()}
            # Finetuning
            log_dir, ft_model = finetune(pt_model, robot, task)
            # Evaluation
            ft_results = evaluate(ft_model, robot, task, episodes=10)
            ft_results = {f"ft_{k}": v for k, v in ft_results.items()}
            improvement = ft_results["ft_success_rate"] - zero_results["zero_success_rate"]
            adapt_results = adaptation_speed(log_dir, robot, task, threshold=0.8, episodes=5)
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
with open("../results/robot_ratio.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print("Experiment complete.")
