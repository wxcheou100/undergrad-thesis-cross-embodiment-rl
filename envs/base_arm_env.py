import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class BaseArmEnv(gym.Env):
    def __init__(self, urdf_path, joint_indices, ee_link, task, render=False):
        super(BaseArmEnv, self).__init__()
        self.urdf_path = urdf_path
        self.joint_indices = joint_indices
        self.ee_link = ee_link
        self.render = render
        if self.render:
            self.client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        self.task = task(self.client)
        self.robot = None
        self.max_dof = 7
        self.max_obs_shape = 2 * self.max_dof + 3 * 6
        self.dof = len(self.joint_indices)
        self.action_space = spaces.Box(low=-0.03, high=0.03, shape=(self.max_dof,), dtype=np.float32)
        # action mask is dependent on the dof of the arm
        self.action_mask = np.zeros(self.max_dof, dtype=np.float32)
        self.action_mask[:self.dof] = 1.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_obs_shape,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.robot = p.loadURDF(
            self.urdf_path,
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=self.client
        )
        if hasattr(self.task, "initial_ee_pos"):
            target_pos = self.task.initial_ee_pos()
            self.set_ee_pos(target_pos)
        joint_pos, joint_vel, ee_pos = self.state()
        self.task.reset(self.robot, self.ee_link, ee_pos)
        task_obs = self.task.observation(ee_pos)
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            ee_pos,
            task_obs
        ])
        info = {}
        return obs.astype(np.float32), info

    def step(self, action):
        # applying the action mask before movement
        action = action * self.action_mask
        action = action[:self.dof]
        joint_pos, joint_vel, ee_pos = self.state()
        for i, joint in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot,
                joint,
                p.POSITION_CONTROL,
                targetPosition=joint_pos[i] + action[i],
                force=200,
                physicsClientId=self.client
            )
        p.stepSimulation(physicsClientId=self.client)
        joint_pos, joint_vel, ee_pos = self.state()
        task_obs = self.task.observation(ee_pos)
        distance, reward, terminated, truncated = self.task.step(ee_pos)
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            ee_pos,
            task_obs
        ])
        info = {"distance": distance}
        return obs, reward, terminated, truncated, info

    def close(self):
        p.disconnect(self.client)

    def set_ee_pos(self, target_pos):
        if self.urdf_path == "franka_panda/panda.urdf" and target_pos[0] > 0.1:
            orn = p.getQuaternionFromEuler([0, 0, 0])
            target_pos[2] += 0.25
        elif self.urdf_path != "franka_panda/panda.urdf" and target_pos[0] <= 0.1:
            orn = p.getQuaternionFromEuler([0, 0, 0])
        else:
            orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        joint_pos = p.calculateInverseKinematics(
            self.robot,
            self.ee_link,
            target_pos,
            orn,
            physicsClientId=self.client
        )
        for i, joint in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot,
                joint,
                joint_pos[i],
                physicsClientId=self.client
            )

    def state(self):
        joint_states = p.getJointStates(self.robot, self.joint_indices, physicsClientId=self.client)
        joint_pos = np.array([s[0] for s in joint_states])
        joint_vel = np.array([s[1] for s in joint_states])
        if len(joint_pos) < self.max_dof:
            pad_size = self.max_dof - len(joint_pos)
            joint_pos = np.concatenate([joint_pos, np.zeros(pad_size)])
            joint_vel = np.concatenate([joint_vel, np.zeros(pad_size)])
        ee_state = p.getLinkState(self.robot, self.ee_link, physicsClientId=self.client)
        ee_pos = np.array(ee_state[0])
        return joint_pos, joint_vel, ee_pos
