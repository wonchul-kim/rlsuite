
from typing import Optional
import gymnasium as gym 
import numpy as np
from rlsuite.envs.m1013.kinematics import fk_delta

class M1013Env(gym.Env):
    def __init__(self):
        
        self.init_joint_position = [0.0, -0.17444443702697754, 1.5700000524520874, 0.0, 1.5700000524520874, 0.0]
        self.joint_position = self.init_joint_position
        self.init_ee_position = [0.4676724374294281, 0.03708246722817421, 0.7366037368774414]
        self.ee_position = self.init_ee_position
        self.current_step = 0

        self.observation_space = gym.spaces.Dict(
            {
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=float),   # [x, y] coordinates
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=float),  # [x, y] coordinates
                "observations": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(len(self.joint_position) + 3,), dtype=float,
                )
            }
        )
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(len(self.joint_position),), dtype=float)
        
        self.target_range = 0.5
        self.goal = None
        
    def get_dist(self, achieved_goal, desired_goal):
        return np.linalg.norm(achieved_goal - desired_goal)
        
    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Gymnasium GoalEnv 호환 보상 함수.
        입력:
        achieved_goal: (..., 3) EE 현재 위치
        desired_goal:  (..., 3) 목표 위치
        info: dict (미사용; 시그니처 유지를 위해 존재)
        반환:
        reward: (...)  # 입력이 (3,)이면 스칼라, (N,3)이면 (N,) numpy.ndarray
        규칙:
        - dense:  -||achieved - desired||
        - sparse:  0 if dist <= goal_threshold else -1
        """
        import numpy as np

        ag = np.asarray(achieved_goal, dtype=np.float32)
        dg = np.asarray(desired_goal,  dtype=np.float32)

        # 브로드캐스팅 허용, 마지막 축(=좌표축) 기준 거리
        dist = np.linalg.norm(ag - dg, axis=-1)

        # if getattr(self, "reward_type", "dense") == "dense":
        #     r = -dist
        # else:  # 'sparse'
        #     thr = float(getattr(self, "goal_threshold", 0.02))
        #     r = -(dist > thr).astype(np.float32)  # <=thr -> 0,  >thr -> -1

        r = -dist

        # 스칼라로 들어오면 스칼라(float32)로, 배치면 (N,) float32로 반환
        return r.astype(np.float32)

    def sample_goal(self):
        goal = self.init_ee_position + self.np_random.uniform(
                    -self.target_range, self.target_range, size=3
                )
        
        return goal
        
    def _get_obs(self):
        obs = {
            'achieved_goal': self.ee_position,
            'desired_goal': self.goal,
            'observations': list(np.concatenate([self.joint_position, np.array([e - g for e, g in zip(self.ee_position, self.goal)])])),
        }
        return obs 
    
    def _get_info(self):
        
        return {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.joint_position = self.init_joint_position
        self.ee_position = self.init_ee_position

        self.goal = self.sample_goal()
        observation = self._get_obs()
        info = self._get_info()
        self.current_step = 0
        
        return observation, info 
    
    def step(self, action):
        # Map the discrete action (0-3) to a movement direction
        
        assert len(action) == self.action_space.shape[0]
        _, ee_position, rpy = fk_delta(action)
        self.ee_position = ee_position
        self.joint_position += action 

        observation = self._get_obs()
        info = self._get_info()

        terminated = False
        if self.get_dist(self.ee_position, self.goal) < 0.02:
            terminated = True 
            info.update({"is_success": True})
        else:
            info.update({"is_success": False})
            
        truncated = False
        if self.current_step > 50:
            truncated = True 
        self.current_step += 1
        
        reward = np.array([[0]]) if terminated else np.array([[-self.get_dist(self.ee_position, self.goal)]])
        
        return observation, reward, terminated, truncated, info        
        

if __name__ == '__main__':
    
    gym.register(
        id="M1013Env-v0",
        entry_point=M1013Env,
        max_episode_steps=100,  # Prevent infinite episodes
    )

    env_name = 'M1013Env-v0'
    env = gym.make(env_name)

    obs, info = env.reset()
    for _ in range(10):
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        dist = np.linalg.norm(obs['achieved_goal']- obs['desired_goal'])
        print(obs)
        print(reward)
        print(dist)
    