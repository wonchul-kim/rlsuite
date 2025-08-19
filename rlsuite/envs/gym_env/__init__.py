import numpy as np
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)


registered_envs = set(gym.envs.registry.keys())
print(registered_envs)

for registered_env in registered_envs:
    if 'Fetch' in registered_env:
        print(registered_env)


# env_name = 'FetchReach-v1'
# env_name = 'FetchReachDense-v3'
env_name = 'FetchReach-v4'
env = gym.make(env_name)
'''
* observation: 
    - (10,)
    - [end_effector_x, end_effector_y, end_effector_z, 
       right_gripper_finger, left_gripper_finger, 
       end_effector_vx, end_effector_vy, end_effector_vz,
       right_gripper_figner_v, left_gripper_finger_z]
       
        - position: m
        - velocity: m/s

* action
    - (4,)
    - [end_effector_dx, end_effector_dy, end_effector_dz, gripper open/close]
    - each value is from -1 to 1
'''

obs, info = env.reset()
for _ in range(10):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    dist = np.linalg.norm(obs['achieved_goal']- obs['desired_goal'])
    print(obs, reward, dist)
 