from envs.panda_env import PandaEnv
from envs.kuka_env import KukaEnv
from envs.xarm6_env import Xarm6Env

from tasks.reach_task import ReachTask
from tasks.push_task import PushTask
from tasks.push_moving_goal_task import PushMovingGoalTask


def make_env(robot: str, task: str, render=False):
    robot_map = {
        "panda": PandaEnv,
        "kuka": KukaEnv,
        "xarm6": Xarm6Env
    }
    if robot not in robot_map:
        raise ValueError(f"Unknown robot: {robot}")

    task_map = {
        "reach": ReachTask,
        "push": PushTask,
        "moving": PushMovingGoalTask
    }
    if task not in task_map:
        raise ValueError(f"Unknown task: {task}")

    return robot_map[robot](
        task=task_map[task],
        render=render
    )
