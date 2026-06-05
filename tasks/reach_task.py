import numpy as np
import pybullet as p


class ReachTask:
    def __init__(self, client, max_steps=800):
        self.client = client
        self.max_steps = max_steps
        self.steps = 0
        self.robot = None
        self.ee_link = None
        self.goal_pos = None
        self.goal = None
        self.prev_distance = 0

    def reset(self, robot, ee_link, ee_pos):
        self.steps = 0
        self.robot = robot
        self.ee_link = ee_link
        # Randomize goal position
        r = np.random.uniform(0.4, 0.5)
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)
        goal_x = ee_pos[0] + r * np.cos(theta)
        goal_y = ee_pos[1] + r * np.sin(theta)
        goal_z = np.random.uniform(0.02, 0.07)
        self.goal_pos = np.array([goal_x, goal_y, goal_z])
        # Load goal
        self.goal = p.loadURDF(
            "cube_small.urdf",
            self.goal_pos,
            physicsClientId=self.client
        )
        p.changeDynamics(self.goal, -1, mass=0)
        self.prev_distance = np.linalg.norm(ee_pos - self.goal_pos)

    def observation(self, ee_pos):
        obj_pos = self.goal_pos.copy()
        rel_ee_goal_pos = self.goal_pos - ee_pos
        rel_ee_obj_pos = obj_pos - ee_pos
        rel_obj_goal_pos = self.goal_pos - obj_pos
        task_obs = np.concatenate([
            obj_pos,
            self.goal_pos,
            rel_ee_goal_pos,
            rel_ee_obj_pos,
            rel_obj_goal_pos
        ])
        return task_obs

    def step(self, ee_pos):
        self.steps += 1
        distance, reward, success = self.reward(ee_pos)
        terminated = success
        truncated = self.steps >= self.max_steps
        return distance, reward, terminated, truncated

    def reward(self, ee_pos):
        distance = np.linalg.norm(self.goal_pos - ee_pos)
        progress = self.prev_distance - distance
        reward = 5 * progress - 3 * distance - 0.01
        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=self.goal,
            linkIndexA=self.ee_link,
            physicsClientId=self.client
        )
        success = (distance < 0.05) or (len(contacts) > 0)
        if success:
            reward += 20
        self.prev_distance = distance
        return distance, reward, success

    @staticmethod
    def initial_ee_pos():
        return np.array([0.1, 0.0, np.random.uniform(0.8, 0.9)])
