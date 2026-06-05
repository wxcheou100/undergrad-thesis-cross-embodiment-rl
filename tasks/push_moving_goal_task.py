import numpy as np
import pybullet as p


class PushMovingGoalTask:
    def __init__(self, client, max_steps=800):
        self.client = client
        self.max_steps = max_steps
        self.steps = 0
        self.robot = None
        self.ee_link = None
        self.object_pos = None
        self.goal_pos = None
        self.object = None
        self.goal = None
        # Parameters for moving goal
        self.goal_speed = 0.0075
        self.goal_direction = np.array([0.0, 0.5, 0.0])
        self.prev_dist_ee_obj = 0
        self.prev_dist_obj_goal = 0

    def reset(self, robot, ee_link, ee_pos):
        self.steps = 0
        self.robot = robot
        self.ee_link = ee_link
        # Randomize object position
        r = np.random.uniform(0.15, 0.2)
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)
        obj_x = ee_pos[0] + r * np.cos(theta)
        obj_y = ee_pos[1] + r * np.sin(theta)
        obj_z = 0.02
        self.object_pos = [obj_x, obj_y, obj_z]
        # Randomize goal position
        r2 = np.random.uniform(0.15, 0.25)
        theta2 = np.random.uniform(-np.pi / 6, np.pi / 6)
        goal_x = obj_x + r2 * np.cos(theta2)
        goal_y = obj_y + r2 * np.sin(theta2)
        goal_z = 0.02
        self.goal_pos = np.array([goal_x, goal_y, goal_z])
        # Load object and goal
        self.object = p.loadURDF(
            "cube_small.urdf",
            self.object_pos,
            physicsClientId=self.client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[1, 0, 0, 0.8]
        )
        self.goal = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.goal_pos
        )
        self.goal_speed = np.random.choice([-1, 1]) * self.goal_speed

    def observation(self, ee_pos):
        obj_pos, _ = p.getBasePositionAndOrientation(self.object, physicsClientId=self.client)
        obj_pos = np.array(obj_pos)
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
        self.move_goal()
        distance, reward, success = self.reward(ee_pos)
        terminated = success
        truncated = self.steps >= self.max_steps
        return distance, reward, terminated, truncated

    def move_goal(self):
        # Move goal position (with boundary)
        self.goal_pos = self.goal_pos + self.goal_direction * self.goal_speed
        if abs(self.goal_pos[1]) >= 0.5:
            self.goal_direction[1] *= -1
        # Update goal
        p.resetBasePositionAndOrientation(self.goal, self.goal_pos, [0, 0, 0, 1], physicsClientId=self.client)

    def reward(self, ee_pos):
        obj_pos, _ = p.getBasePositionAndOrientation(self.object, physicsClientId=self.client)
        obj_pos = np.array(obj_pos)
        dist_ee_obj = np.linalg.norm(obj_pos - ee_pos)
        dist_obj_goal = np.linalg.norm(self.goal_pos - obj_pos)
        progress_ee_obj = self.prev_dist_ee_obj - dist_ee_obj
        progress_obj_goal = self.prev_dist_obj_goal - dist_obj_goal
        reward = 2 * progress_ee_obj - dist_ee_obj + 5 * progress_obj_goal - 2 * dist_obj_goal - 0.01
        contacts = p.getContactPoints(
            bodyA=self.robot,
            bodyB=self.object,
            physicsClientId=self.client
        )
        contact_reward = 0.0
        if len(contacts) > 0:
            contact_reward = 0.1
        for contact in contacts:
            if contact[3] == self.ee_link:
                contact_reward = 0.5
                break
        reward += contact_reward
        success = dist_obj_goal < 0.05
        if success:
            reward += 20
        self.prev_dist_ee_obj = dist_ee_obj
        self.prev_dist_obj_goal = dist_obj_goal
        return dist_obj_goal, reward, success

    @staticmethod
    def initial_ee_pos():
        return np.array([0.2, 0.0, np.random.uniform(0.05, 0.15)])
