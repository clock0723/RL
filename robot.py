import os
import numpy as np
import mujoco
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces

if not hasattr(mujoco, "mju_quat2Euler"):
    def _mju_quat2Euler(quat):
        mat = np.empty(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=np.float64))
        mat = mat.reshape(3, 3)

        roll = np.arctan2(mat[2, 1], mat[2, 2])
        pitch = np.arcsin(np.clip(-mat[2, 0], -1.0, 1.0))
        yaw = np.arctan2(mat[1, 0], mat[0, 0])

        return np.array([roll, pitch, yaw], dtype=np.float32)

    mujoco.mju_quat2Euler = _mju_quat2Euler

class Env(MujocoEnv):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        model_path = os.path.join(os.path.dirname(__file__), "robot.xml")
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        super().__init__(
            model_path=model_path,
            frame_skip=5,
            observation_space=observation_space,
            render_mode=render_mode
        )
        self.init_x = 0.0

        names = self.model.names
        if isinstance(names, (bytes, bytearray)):
            names = names.decode("ascii")
        else:
            names = "".join(chr(c) for c in names)
        self._body_name_to_id = {}
        for i in range(self.model.nbody):
            start = int(self.model.name_bodyadr[i])
            end = names.index("\x00", start)
            self._body_name_to_id[names[start:end]] = i
        self.left_knee_bodyid = self._body_name_to_id["left_leg"]
        self.right_knee_bodyid = self._body_name_to_id["right_leg"]

    def get_obs(self):
        d = self.data
        body_pos = d.qpos[0:3]
        body_angle = mujoco.mju_quat2Euler(d.qpos[3:7])
        body_vel = d.qvel[0:3]
        body_anglevel = d.qvel[3:6]
        left_leg_pos = d.qpos[7:10]
        left_leg_vel = d.qvel[6:9]
        right_leg_pos = d.qpos[10:13]
        right_leg_vel = d.qvel[9:12]
        obs = np.concatenate([
            body_pos, body_angle, body_vel, body_anglevel,
            left_leg_pos, left_leg_vel, right_leg_pos, right_leg_vel
        ])
        return obs.astype(np.float32)

    def reward(self, obs, action):
        # 1. 主要目标：站稳，惩罚俯仰角和反转角
        body_angle = mujoco.mju_quat2Euler(self.data.qpos[3:7])
        roll = body_angle[0]
        pitch = body_angle[1]
        stand_reward = 1.0 - (roll ** 2 + pitch ** 2)

        # 2. 动作成本（重要！）
        ctrl_cost = 0.01 * np.sum(np.square(action))  # 权重调小一点

        # 3. 生存奖励（每步固定）
        survive_reward = 0.5

        # 4. 前进奖励（可选，如果希望移动）
        #forward_reward = 2.0 * (self.data.qpos[0] - self.init_x) / self.dt  # x 方向的前进奖励
        
        

        # 总奖励
        reward = stand_reward + survive_reward - ctrl_cost + forward_reward
        return reward

    def _get_knee_height(self):
        d = self.data
        left_knee_z = d.xpos[self.left_knee_bodyid, 2]
        right_knee_z = d.xpos[self.right_knee_bodyid, 2]
        return min(left_knee_z, right_knee_z)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self.get_obs()
        reward = self.reward(obs, action)
        terminated = self.check_terminated()
        truncated = False
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, {}

    def check_terminated(self):
        body_angle = mujoco.mju_quat2Euler(self.data.qpos[3:7])
        roll = body_angle[0]
        pitch = body_angle[1]
        if abs(roll) > 0.5 or abs(pitch) > 0.5:
            return True
        if self.data.qpos[2] < 0.5:
            return True
        if self._get_knee_height() < 0.4:
            return True
        return False

    def reset_model(self):
        noise = 0.002
        qpos = self.init_qpos + self.np_random.uniform(low=-noise, high=noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-noise, high=noise, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.init_x = self.data.qpos[0]
        return self.get_obs()