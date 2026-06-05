import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium import spaces

class Env(MujocoEnv):
    metadata = {"render_modes":["human", "rgb_array"], "render_fps": 60}

    def  __init__(self):
        MujocoEnv.__init__(self, "robot.xml", 5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
    
    def get_obs(self):
        d=self.data
        body_pos=d.qpos[0:3]
        body_angle=np.zeros(3)
        mujoco.mujoco.mju_quat2euler(body_angle, d.qpos[3:7])
        body_vel=d.qvel[0:3]
        body_anglevel=d.qvel[3:7]
        left_leg_pos=d.qpos[7:10]
        left_leg_vel=d.qvel[7:10]
        right_leg_pos=d.qpos[10:13]
        right_leg_vel=d.qvel[10:13]
        distance=body_pos[0]-self.init_x
        obs=np.concatenate([body_pos, body_angle, body_vel, body_anglevel, left_leg_pos, left_leg_vel, right_leg_pos, right_leg_vel])
        return obs.astype(np.float32)
    
    def reward(self, obs):
        d=self.data
        body_angle=np.zeros(3)
        mujoco.mujoco.mju_quat2euler(body_angle, d.qpos[3:7])
        roll=body_angle[0]
        pitch=body_angle[1]
        reward_roll=np.clip(np.abs(roll)-0.25, 0, None)
        reward_pitch=np.clip(np.abs(pitch)-0.25, 0, None)
        r1=-reward_roll*5-reward_pitch*5 #平衡奖励
        distance=obs[0]-self.init_x
        r2=distance #前进奖励
        r3=0.1 #生存奖励
        reward=r1+r2+r3  
        info={'balance_reward': r1, 'distance_reward': r2, 'survival_reward': r3}
        return reward
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs=self.get_obs()
        reward=self.reward(obs)
        terminated=self_check_termination(obs)
        truncated=False     
        info={}
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, truncated, info
    
    def check_terminated(self):
        d=self.data
        body_pos=d.qpos[0:3]
        body_angle=np.zeros(3)
        mujoco.mujoco.mju_quat2euler(body_angle, d.qpos[3:7])
        roll=body_angle[0]
        pitch=body_angle[1]
        if abs(roll)>0.5 or abs(pitch)>0.5:
            return True
        if body_pos[2]<0.2:
            return True
        return False
    
    def reset(self):
        noise=0.02
        qpos=self.init_qpos + self.np_random.uniform(low=-noise, high=noise, size=self.model.nq)
        qvel=self.init_qvel + 0
        self.set_state(qpos, qvel)
        self.check_terminated
        obs=self.get_obs()
        return obs, {}
