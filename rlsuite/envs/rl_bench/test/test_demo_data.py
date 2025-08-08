


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
            'amount': 10,
        },
        'renderer': {
            'use': False,
            'render_mode': 'rgb_array',
        },
        'step_length': 100,
        'custom_reward': 'reshape_reward_function'
            
    }
    env = RLBenchEnv(config)
    
    desc, obs = env.reset()
    done = False 

    demos = env.get_demos()
    
    
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    plt.figure()
    
    for epi_idx, espisode_demo in enumerate(demos):
        observations = espisode_demo['observations']
        actions = espisode_demo['actions']
        rewards = espisode_demo['rewards']
        terminals = espisode_demo['terminals']
        
        reward_list, dist_list = [], []
        for idx in tqdm(range(len(observations)), desc=f'epsidoe {epi_idx}'):
            target_position = observations[idx]['task_low_dim_state']
            ee_position = observations[idx]['gripper_pose'][:3]
            
            dist = np.linalg.norm(ee_position - target_position)

            dist_list.append(dist)
            reward_list.append(rewards[idx])
            
        plt.plot(dist_list, label=f'dist_{epi_idx}')
        plt.plot(reward_list, label=f'reward_{epi_idx}')
        
    plt.legend()
    plt.savefig('/HDD/etc/outputs/dist_reward.png')
    plt.close()
                   
            
            
            
        
            
            
    
    

    

