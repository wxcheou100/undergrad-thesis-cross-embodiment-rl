import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.experiment_config import CONFIG

cfg = CONFIG
df = pd.read_csv("robot_ratio.csv")
tasks = cfg["tasks"]
robots = cfg["robots"]
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
for ax, robot in zip(axes, robots):
    # Filter dataframe for current robot
    task_df = df[df["robot"] == robot]
    # Base x positions for tasks
    x = np.arange(len(tasks))
    for i, ratio in enumerate(ratios):
        ratio_df = task_df[task_df["robot_ratio"] == ratio]
        zero_vals = []
        ft_vals = []
        # Collect values in fixed task order
        for task in tasks:
            row = ratio_df[ratio_df["task"] == task]
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
            label=ratio if robot == "panda" else ""
        )
    # subplot formatting
    if robot == "xarm6":
        robot_name = "xArm6"
    else:
        robot_name = robot.capitalize()
    ax.set_title(robot_name, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.capitalize() for t in tasks],
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
    title="Robot Ratio",
    loc="upper center",
    ncol=5,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.98)
)
fig.suptitle(
    "Robot Ratio Experiment",
    fontsize=16,
    y=1.03
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
# saving the figure
plt.savefig(
    "robot_ratio_success.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

summary = df.groupby("robot_ratio").agg({
    "ft_avg_reward": "mean",
    "ft_avg_final_distance": "mean"
})
# print(summary)
summary.to_csv("robot_ratio_appendix_reward.csv")

adapt_df = df[df["adapt_timesteps_to_threshold"].notna()]
adapt_df = adapt_df[
    [
        "robot_ratio",
        "task_ratio",
        "robot",
        "task",
        "adapt_timesteps_to_threshold"
    ]
]
# print(adapt_df)
adapt_df.to_csv(
    "robot_ratio_appendix_adapt.csv",
    index=False
)
