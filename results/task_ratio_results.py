import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment_config import CONFIG

cfg = CONFIG
df = pd.read_csv("task_ratio.csv")
robots = cfg["robots"]
robot_names = ["Panda", "Kuka", "xArm6"]
tasks = cfg["tasks"]
ratios = [
    "[100, 0]",
    "[75, 25]",
    "[50, 50]",
    "[25, 75]",
    "[0, 100]"
]
colors = [
    "tab:red",
    "tab:orange",
    "tab:green",
    "tab:blue",
    "tab:purple"
]
bar_width = 0.07

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, task in zip(axes, tasks):
    # Filter dataframe for current task
    task_df = df[df["task"] == task]
    # Base x positions for robots
    x = np.arange(len(robots))
    for i, ratio in enumerate(ratios):
        ratio_df = task_df[task_df["task_ratio"] == ratio]
        zero_vals = []
        ft_vals = []
        # Collect values in fixed robot order
        for robot in robots:
            row = ratio_df[ratio_df["robot"] == robot]
            zero_vals.append(
                row["zero_success_rate"].values[0] * 100
            )
            ft_vals.append(
                row["ft_success_rate"].values[0] * 100
            )
        # Offset positioning
        offset = (i - 2) * 2 * bar_width
        # zero-shot bars (light color)
        ax.bar(
            x + offset,
            zero_vals,
            width=bar_width,
            color=colors[i],
            alpha=0.33
        )
        # fine-tune bars (dark color)
        ax.bar(
            x + offset + bar_width,
            ft_vals,
            width=bar_width,
            color=colors[i],
            alpha=1.0,
            label=ratio if task == "reach" else ""
        )
    # subplot formatting
    ax.set_title(task.capitalize(), fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        robot_names,
        fontsize=11
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

# global legend and title
fig.legend(
    title="Task Ratio",
    loc="upper center",
    ncol=5,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.98)
)
fig.suptitle(
    "Task Ratio Experiment",
    fontsize=16,
    y=1.03
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
# saving the figure
plt.savefig(
    "task_ratio_success.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

summary = df.groupby("task_ratio").agg({
    "ft_avg_reward": "mean",
    "ft_avg_final_distance": "mean"
})
# print(summary)
summary.to_csv("task_ratio_appendix_reward.csv")

adapt_df = df[df["adapt_timesteps_to_threshold"].notna()]
adapt_df = adapt_df[
    [
        "task_ratio",
        "robot_ratio",
        "task",
        "robot",
        "adapt_timesteps_to_threshold"
    ]
]
# print(adapt_df)
adapt_df.to_csv(
    "task_ratio_appendix_adapt.csv",
    index=False
)
