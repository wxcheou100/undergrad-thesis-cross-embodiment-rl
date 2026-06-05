from envs.base_arm_env import BaseArmEnv


class Xarm6Env(BaseArmEnv):
    def __init__(self, task, render=False):
        urdf_path = "xarm/xarm6_robot.urdf"
        joint_indices = [1, 2, 3, 4, 5, 6]
        ee_link = 6
        super().__init__(urdf_path, joint_indices, ee_link, task, render=render)
