# import gymnasium as gym

# from stable_baselines3 import PPO

# env = gym.make("CartPole-v1", render_mode="human")

# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10_000)

# vec_env = model.get_env()
# obs = vec_env.reset()
# for i in range(1000):
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = vec_env.step(action)
#     vec_env.render()
#     # VecEnv resets automatically
#     # if done:
#     #   obs = env.reset()

# env.close()


import gymnasium as gym 
from stable_baselines3 import PPO
from rlsuite.envs.rl_bench.rlbench_env import RLBenchEnv

import time 
config = {
    'env':{
        'obs': {
            'set_all': False,
            'set_all_high_dim': False,
            'set_all_low_dim': True,
            
        },
        'arm_max_velocity': 2.0,
        'arm_max_acceleration': 8.0,
        'dataset_root': '/HDD/etc/rlbench_demo/reach_target_100',
    },
    'observations':{
        'low_dim_obs': ['joint_positions', 'joint_velocities', 'gripper_open'],
        'high_dim_obs': {
            'rgb': [],
            'mask': [],
            'depth': [],
        },
        'frame_stack': 1,
    },
    'demo': {
        'amount': 100,
    },
    'renderer': {
        'use': False,
        'render_mode': 'rgb_array',
    },
    'step_length': 100,
        
}
env = RLBenchEnv(config)

desc, obs = env.reset()
done = False 

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# for step_idx in range(1e5):
    
#     episode_step_idx = 1
#     while not done:
        
#         action, _states = model.
        
#         obs, reward, done, info = env.step(env.sample_action())
#         episode_step_idx += 1

