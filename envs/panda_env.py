from envs.base_arm_env import BaseArmEnv


class PandaEnv(BaseArmEnv):
    def __init__(self, task, render=False):
        urdf_path = "franka_panda/panda.urdf"
        joint_indices = [0, 1, 2, 3, 4, 5, 6]
        ee_link = 11
        super().__init__(urdf_path, joint_indices, ee_link, task, render=render)
