
if __name__ == '__main__':
    import argparse 
    import os.path as osp
    import os
    from pathlib import Path
    from tqdm import tqdm
    
    from rlsuite.envs.rl_bench.env import make
    from rlsuite.envs.rl_bench.utils.video_recorder import VideoRecorder


    cfg = argparse.ArgumentParser().parse_args()
    cfg.task_name = 'take_lid_off_saucepan'
    cfg.episode_length = 10000
    cfg.frame_stack = 8
    cfg.dataset_root = '/HDD/etc/rlbench_demo'
    cfg.num_demos = 100
    cfg.arm_max_velocity = 2.0
    cfg.arm_max_acceleration = 8.0
    cfg.camera_shape = [84,84]
    cfg.camera_keys = ['front', 'wrist', 'left_shoulder', 'right_shoulder']
    cfg.state_keys = ['joint_positions', 'gripper_open']
    cfg.renderer = 'opengl3'
    
    cfg.output_dir = '/HDD/etc/rlbench_demo/video'
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    env = make(
            cfg.task_name,
            cfg.episode_length,
            cfg.frame_stack,
            cfg.dataset_root,
            cfg.arm_max_velocity,
            cfg.arm_max_acceleration,
            cfg.camera_shape,
            cfg.camera_keys,
            cfg.state_keys,
            cfg.renderer,
        )
    
    print(f"rgb_raw_observation_spec: {env.rgb_raw_observation_spec()}")
    print(f"low_dim_raw_observation_spec: {env.low_dim_raw_observation_spec()}")
    print(f"action_spec: {env.action_spec()}")
    
    demos = env.get_demos(cfg.num_demos)
    recorder = VideoRecorder(Path(cfg.output_dir))
    for idx, episode in enumerate(demos):
        _time_step = env.reset()
        recorder.init(env, enabled=True)
        frames = []
        for time_step in tqdm(episode):
            _time_step = env.step(time_step.action)
            frames.append(time_step.rgb_obs)
            recorder.record(env)
            
        # imageio.mimsave(f"episode_{idx}.gif", frames, fps=20)
        recorder.save(f"episode_{idx}_.gif")
        
    
'''
CUDA_VISIBLE_DEVICES=1 DISPLAY=:1.0 python dataset_generator.py --save_path=/HDD/etc/rlbench_demo --image_size 84 84 --renderer opengl3 --episodes_per_task 10000 --variations 1 --processes 1 --tasks reach_target --arm_max_velocity 2.0 --arm_max_acceleration 8.0
'''