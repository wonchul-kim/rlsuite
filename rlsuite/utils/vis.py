import pandas as pd 
import numpy as np 
from collections import OrderedDict
import pickle

filepath = '/HDD/etc/outputs/tqc/tqc/M1013Env-v0_7/trajectories/trajectories.pkl'

with open(filepath, "rb") as f:   # adjust path if needed
    trajectories = pickle.load(f)

print(type(trajectories))                # dict
print(trajectories.keys())               # e.g. dict_keys(['trajectory_1', 'trajectory_2'])
print(trajectories["trajectory_1"].keys())  # e.g. dict_keys(['observations','dones','actions','rewards'])



import gymnasium_robotics
import gymnasium as gym
gym.register_envs(gymnasium_robotics)
from robot_sim.robots.doosan.env import M1013Env
gym.register(
    id="M1013Env-v0",
    entry_point=M1013Env,
    # max_episode_steps=100,  # Prevent infinite episodes
)

from robot_sim.robots.doosan.env_v2 import M1013EnvV2
gym.register(
    id="M1013Env-v1",
    entry_point=M1013EnvV2,
    # max_episode_steps=100,  # Prevent infinite episodes
)

env = gym.make('M1013Env-v0')
obs, info = env.reset()

for key, val in trajectories.items():
    traj_obervations = val['observations']
    traj_dones = val['dones']
    traj_actions = val['actions']
    traj_rewards = val['rewards']
    
    traj_coordinates = {'x': [], 'y': [], 'z': []}
    coordinates = {'x': [], 'y': [], 'z': []}
    for traj_o, traj_d, traj_a, traj_r in zip(traj_obervations, traj_dones, traj_actions, traj_rewards):
        
        traj_coordinates['goal'] = traj_o['desired_goal']
        obs, reward, done, truncated, info = env.step(traj_a.flatten())
        
        print("traj_o['achieved_goal']: ", traj_o['achieved_goal'])
        print("obs['achieved_goal']: ", obs['achieved_goal'])
        print(np.allclose(traj_o['achieved_goal'], obs['achieved_goal']))
        
        traj_coordinates['x'].append(traj_o['achieved_goal'][0][0])
        traj_coordinates['y'].append(traj_o['achieved_goal'][0][1])
        traj_coordinates['z'].append(traj_o['achieved_goal'][0][2])
        
        coordinates['x'].append(obs['achieved_goal'][0])
        coordinates['y'].append(obs['achieved_goal'][1])
        coordinates['z'].append(obs['achieved_goal'][2])
        
    import numpy as np
    import matplotlib.pyplot as plt

    # Example: traj_coordinates = {"x": [...], "y": [...], "z": [...]}

    # Convert to numpy arrays (and sanity‑check same length)
    x = np.asarray(traj_coordinates["x"], dtype=float)
    y = np.asarray(traj_coordinates["y"], dtype=float)
    z = np.asarray(traj_coordinates["z"], dtype=float)
    _x = np.asarray(coordinates["x"], dtype=float)
    _y = np.asarray(coordinates["y"], dtype=float)
    _z = np.asarray(coordinates["z"], dtype=float)
    n = min(len(x), len(y), len(z))
    x, y, z = x[:n], y[:n], z[:n]

    # 3D line + scatter
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)        # line through the points
    ax.plot(_x, _y, _z)        # line through the points
    ax.scatter([traj_coordinates['goal'][0][0]],
               [traj_coordinates['goal'][0][1]],
               [traj_coordinates['goal'][0][2]])
    ax.scatter(x, y, z, s=12)  # points

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trajectory")

    # Make axes roughly equal scale
    def _range(a): 
        return float(np.ptp(a)) if np.ptp(a) > 0 else 1.0
    ax.set_box_aspect([_range(x), _range(y), _range(z)])

    plt.savefig(f'/HDD/etc/outputs/tqc/tqc/M1013Env-v0_7/trajectories/{key}.png')
