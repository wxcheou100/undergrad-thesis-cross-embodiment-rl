import os
import csv
import time
from training.pretrain import pretrain
from training.finetune import finetune
from evaluation.evaluate import evaluate
from evaluation.adaptation_speed import adaptation_speed
from utils.experiment_config import CONFIG

cfg = CONFIG
robot_ratios = [
    [100, 0],
    [75, 25],
    [50, 50],
    [25, 75],
    [0, 100]
]
task_ratio = [50, 50]
results = []
total_ft_jobs = len(robot_ratios) * len(cfg["robots"]) * len(cfg["tasks"])
ft_job = 0
overall_start = time.time()

for robot_ratio in robot_ratios:
    # Pretraining
    print("\n=====================================================================")
    print(f"Pretraining on robot ratio {robot_ratio} begins.")
    pt_start = time.time()
    pt_model = pretrain(robot_ratio, task_ratio)
    pt_time = time.time() - pt_start
    print(f"Pretraining on robot ratio {robot_ratio} completed in {pt_time/60:.2f} minutes.")
    print("=====================================================================")
    for robot in cfg["robots"]:
        for task in cfg["tasks"]:
            zero_results = evaluate(pt_model, robot, task, cfg["eval_episodes"])
            zero_results = {f"zero_{k}": v for k, v in zero_results.items()}
            # Finetuning
            ft_job += 1
            print("\n---------------------------------------------------------------------")
            print(f"Finetuning job {ft_job}/{total_ft_jobs} begins.")
            print(f"Robot ratio {robot_ratio} with robot {robot} and task {task}.")
            ft_start = time.time()
            log_dir, ft_model = finetune(pt_model, robot, task)
            ft_time = time.time() - ft_start
            total_time = time.time() - overall_start
            avg_time = total_time/ft_job
            remaining_time = avg_time * (total_ft_jobs - ft_job)
            print(f"Finetuning job {ft_job}/{total_ft_jobs} completed in {ft_time/60:.2f} minutes.")
            print(f"Elapsed time: {total_time/60:.2f} minutes.")
            print(f"Estimated remaining time: {remaining_time/60:.2f} minutes.")
            print("---------------------------------------------------------------------")
            # Evaluation
            ft_results = evaluate(ft_model, robot, task, cfg["eval_episodes"])
            ft_results = {f"ft_{k}": v for k, v in ft_results.items()}
            improvement = ft_results["ft_success_rate"] - zero_results["zero_success_rate"]
            adapt_results = adaptation_speed(log_dir, robot, task, cfg["threshold"], cfg["adapt_episodes"])
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
total_time = time.time() - overall_start
print(f"Experiment completed in {total_time/3600:.2f} hours.")
