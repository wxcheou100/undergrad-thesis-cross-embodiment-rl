import random
import gymnasium as gym
from utils.env_wrapper import make_env


class MixedArmEnv(gym.Env):
    def __init__(self, robots, robot_ratio, tasks, task_ratio, render=False):
        super(MixedArmEnv, self).__init__()
        self.robots = robots
        self.robot_ratio = robot_ratio
        self.tasks = tasks
        self.task_ratio = task_ratio
        self.render = render
        self.env = None
        # temporary environment to inherit action and observation space
        temp_env = make_env(self.robots[0], self.tasks[0])
        self.action_space = temp_env.action_space
        self.observation_space = temp_env.observation_space
        temp_env.close()

    def reset(self, seed=None, options=None):
        if self.env is not None:
            self.env.close()
        robot = self.sample_robot()[0]
        task = self.sample_task()[0]
        self.env = make_env(robot, task, render=self.render)
        obs, info = self.env.reset(seed=seed, options=options)
        info["robot"] = robot
        info["task"] = task
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env is not None:
            self.env.close()

    def sample_robot(self):
        return random.choices(
            self.robots,
            weights=self.robot_ratio,
            k=1
        )

    def sample_task(self):
        return random.choices(
            self.tasks,
            weights=self.task_ratio,
            k=1
        )
