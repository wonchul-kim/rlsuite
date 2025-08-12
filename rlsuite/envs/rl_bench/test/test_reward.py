


if __name__ == '__main__':
    from rlsuite.envs.rl_bench.rlbench_env import RLBenchEnv
    import numpy as np
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
            'dataset_root': '/HDD/etc/rlbench_demo',
            'headless': True,
            'shaped_rewards': True,
        },
        'observations':{
            'low_dim_obs': ['joint_positions', 'joint_velocities', 'relative_position', 'gripper_open'],
            'high_dim_obs': {
                'rgb': [],
                'mask': [],
                'depth': [],
            },
            'frame_stack': 1,
        },
        'demo': {
            'amount': 1,
        },
        'renderer': {
            'use': False,
            'render_mode': 'rgb_array',
        },
        'step_length': 100,
        # 'custom_reward': 'reshape_reward_function'
        'custom_reward': None,
            
    }
    env = RLBenchEnv(config)
    
    desc, obs = env.reset()
    done = False 
    
    
    import matplotlib.pyplot as plt
    import math
    plt.figure()

    idx = 1
    dist_list, reward_list = [], []
    while not done:
        tic = time.time()
        obs, reward, done, info = env.step(env.sample_action())
        # print(obs, reward, terminate, info)
        print(f"step {idx}: {time.time() - tic}")
        idx += 1
        
        dist = np.linalg.norm(obs[-4:-1])
        dist_list.append(dist)
        reward_list.append(reward)
        
    plt.plot(dist_list, label=f'dist')
    plt.plot(reward_list, label=f'reward')
        
    plt.legend()
    plt.savefig('/HDD/etc/outputs/dist_reward.png')
    plt.close()
                   
            
            
            
        
            
            
    
    

    

