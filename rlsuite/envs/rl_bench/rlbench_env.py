import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget
from rlbench.observation_config import ObservationConfig
from rlbench.action_modes.action_mode import JointPosition
from rlbench.utils import name_to_task_class
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pyrep.const import RenderMode

from typing import Dict, Union
from gymnasium import spaces
from collections import deque 


def reshape_reward_function(obs, done, reward):
    gripper_position = obs.gripper_pose[:3]
    target_position = obs.task_low_dim_state
    dist = np.linalg.norm(gripper_position - target_position)
    alpha = 4
    dist_reward = np.exp(-alpha*dist)
    max_dist = 0.85
    reward += 1. - (dist/max_dist) + dist_reward
    # reward = np.clip(reward, 0., 2.)
    
    if done:
        reward += 5
        
    return reward
        
class RLBenchEnv:
    def __init__(self, config):
        self._config = config
        self._launch()
        if self._config['renderer']['use']:
            self._add_gym_camera()
        self._initialize()
        self._set_spaces()
        
    @property
    def action_shape(self):
        return self._env.action_shape
    
    def sample_action(self, min_action=-1., max_action=1.):
        a = np.random.normal(size=(self.action_shape))
        a = np.clip(a, min_action, max_action)
        
        return a
    
    def _launch(self):
        
        obs_config = ObservationConfig()
        obs_config.set_all(self._config['env']['obs']['set_all'])
        obs_config.set_all_high_dim(self._config['env']['obs']['set_all_high_dim'])
        obs_config.set_all_low_dim(self._config['env']['obs']['set_all_low_dim'])
        arm_max_velocity = self._config['env']['arm_max_velocity']
        arm_max_acceleration = self._config['env']['arm_max_acceleration']
        dataset_root = '/HDD/etc/outputs/rlsuite/rlbench'
        task_name = 'reach_target'
        import os
        os.makedirs(os.path.join(dataset_root), exist_ok=True)
        
        action_mode = MoveArmThenGripper(
                        arm_action_mode=JointPosition(False),
                        gripper_action_mode=Discrete()
                    )
        self._env = Environment(action_mode, 
                        arm_max_velocity=arm_max_velocity,
                        arm_max_acceleration=arm_max_acceleration,
                        obs_config=obs_config,
                        dataset_root=self._config['env']['dataset_root'],
                        headless=self._config['env']['headless'])
        self._env.launch()

        self._task = self._env.get_task(name_to_task_class(task_name))
        
    def _add_gym_camera(self):
        if self._config['renderer']['render_mode'] is not None:
            # Add the camera to the scene
            cam_placeholder = Dummy("cam_cinematic_placeholder")
            self._gym_cam = VisionSensor.create([320, 192])
            self._gym_cam.set_pose(cam_placeholder.get_pose())
            if self._config['renderer']['render_mode'] == "human":
                self._gym_cam.set_render_mode(RenderMode.OPENGL3_WINDOWED)
            else:
                self._gym_cam.set_render_mode(RenderMode.OPENGL3) 

    def _delete_gym_camera(self):
        self._gym_cam.remove()
        del self._gym_cam

    def render(self, mode="rgb_array") -> Union[None, np.ndarray]:
        if mode != self._config['renderer']['render_mode']:
            raise ValueError(
                "The render mode must match the render mode selected in the "
                'constructor. \nI.e. if you want "human" render mode, then '
                "create the env by calling: "
                'gym.make("reach_target-state-v0", render_mode="human").\n'
                "You passed in mode %s, but expected %s." % (mode, self._config['renderer']['render_mode'])
            )
        if mode == "rgb_array":
            frame = self._gym_cam.capture_rgb()
            frame = np.clip((frame * 255.0).astype(np.uint8), 0, 255)
            return frame
    
    def _initialize(self):
        # Create deques for frame stacking
        self._low_dim_obses = deque([], maxlen=self._config['observations']['frame_stack'])
        self._frames = {
            camera_key: deque([], maxlen=self._config['observations']['frame_stack'])
            for camera_key in self._config['observations']['high_dim_obs']['rgb']
        }
        self._num_steps = 0
        self._reward_per_episode = 0
        
        
    def _set_spaces(self):
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._env.action_shape,
            dtype=np.float32
        )
        
        description, obs = self.reset()
        
        self.low_dim_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs.shape[0]*self._config['observations']['frame_stack'],), 
            dtype=np.float32
        ) 
        # self.high_dim_observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(len(self._camera_keys), 3 * self._frame_stack, *self._camera_shape),
        #     dtype=np.uint8,
        # )
        # self.rgb_raw_observation_space = spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(len(self._camera_keys), 3, *self._camera_shape),
        #     dtype=np.uint8,
        # ) 

        # Set default action stats, which will be overridden by demonstration action stats
        # Required for a case we don't use demonstrations
        action_min = (
            -np.ones(self.action_space.shape, dtype=self.action_space.dtype) * 0.2
        )
        action_max = (
            np.ones(self.action_space.shape, dtype=self.action_space.dtype) * 0.2
        )
        action_min[-1] = 0
        action_max[-1] = 1
        self._action_stats = {"min": action_min, "max": action_max}
        
    def reset(self, **kwargs):
        # Clear deques used for frame stacking
        self._num_steps = 0
        self._reward_per_episode = 0
        self._low_dim_obses.clear()
        for frames in self._frames.values():
            frames.clear()
            
        description, obs = self._task.reset(**kwargs)
        obs = self._extract_obs(obs)
        obs = obs['low_dim_obs']
        
        return description, obs    
    
    def step(self, action):
        obs, reward, terminate = self._task.step(action)
        self._num_steps += 1
        
        if self._num_steps >= self._config['step_length']:
            truncated = True 
        else:
            truncated = False 
            
        if terminate or truncated:
            done = 1
        else:
            done = 0
            
        if self._config['custom_reward']:
            reward = reshape_reward_function(obs, done, reward)
        
        self._reward_per_episode += reward
        obs = self._extract_obs(obs)
        obs = obs['low_dim_obs']
        
        
        return obs, reward, done, {"num_steps": self._num_steps, "total_reward": self._reward_per_episode}
        
    def get_relative_position(self, obs):
        return obs['task_low_dim_state'] - obs['gripper_pose'][:3]
        
    def _extract_obs(self, obs) -> Dict[str, np.ndarray]:
        obs = vars(obs)
        out = dict()

        # Get low-dimensional state with stacking
        if len(self._config['observations']['low_dim_obs']) != 0:
            # low_dim_obs = np.hstack(
            #     [obs[key] for key in self._config['observations']['low_dim_obs']], dtype=np.float32
            # )
            low_dim_obs = []
            for key in self._config['observations']['low_dim_obs']:
                if key in obs:
                    low_dim_obs.append(obs[key])
                else:
                    if key == 'relative_position':
                        low_dim_obs.append(self.get_relative_position(obs))
                    else:
                        raise NotImplementedError(f"There is no such observation key: {key} at {obs.keys()}")
            
            low_dim_obs = np.hstack(low_dim_obs, dtype=np.float32)
            if len(self._low_dim_obses) == 0:
                for _ in range(self._config['observations']['frame_stack']):
                    self._low_dim_obses.append(low_dim_obs)
            else:
                self._low_dim_obses.append(low_dim_obs)
            out["low_dim_obs"] = np.concatenate(list(self._low_dim_obses), axis=0)
        else:
            out["low_dim_obs"] = np.array([])
            
        # Get rgb observations with stacking
        if len(self._config['observations']['high_dim_obs']['rgb']) != 0:
            for camera_key in self._config['observations']['high_dim_obs']['rgb']:
                pixels = obs[f"{camera_key}_rgb"].transpose(2, 0, 1).copy()
                if len(self._frames[camera_key]) == 0:
                    for _ in range(self._config['observations']['frame_stack']):
                        self._frames[camera_key].append(pixels)
                else:
                    self._frames[camera_key].append(pixels)
            out["high_dim_obs"] = np.stack(
                [
                    np.concatenate(list(self._frames[camera_key]), axis=0)
                    for camera_key in self._config['observations']['high_dim_obs']['rgb']
                ],
                0,
            )
        else:
            out['high_dim_obs'] = np.array([])
            
        out['task_low_dim_state'] = obs['task_low_dim_state']
        out['gripper_pose'] = obs['gripper_pose']
        
        return out
        
        
    def get_demos(self, modify=True):
        
        live_demos = not self._config['env']['dataset_root']
        
        raw_demos = self._task.get_demos(self._config['demo']['amount'], 
                                         live_demos=live_demos)
        demos = []
        for episode_raw_demo in raw_demos:
            
            episode_demo = self.modify_raw_demos(episode_raw_demo)
            
            if episode_demo is not None:
                demos.append(episode_demo)
            else:
                print("Skipping episode-demo for large delta action")
                
        # # override action stats with demonstration-based stats
        # self._action_stats = self.extract_action_stats(demos)
        # # rescale actions with action stats
        # demos = [self.rescale_demo_actions(demo) for demo in demos]
        
        return demos
        
    def extract_delta_joint_action(self, obs, next_obs):
        action = np.concatenate(
            [
                (
                    next_obs.misc["joint_position_action"][:-1] - obs.joint_positions
                    if "joint_position_action" in next_obs.misc
                    else next_obs.joint_positions - obs.joint_positions
                ),
                [1.0 if next_obs.gripper_open == 1 else 0.0],
            ]
        ).astype(np.float32)
        return action


    def modify_raw_demos(self, episode_raw_demo):
        episode_demo = {'observations': [],
                        'actions': [],
                        'rewards': [],
                        'terminals': []
                }
        for step_idx, step_raw_demo in enumerate(episode_raw_demo):
            
            obs = self._extract_obs(step_raw_demo)
            reward = 0
            terminal = 0
            
            if step_idx == 0:
                action = np.zeros(self.action_shape, dtype=self.action_space.dtype)
            else:
                prev_obs = episode_raw_demo[step_idx - 1]
                action = self.extract_delta_joint_action(prev_obs, step_raw_demo)
                if np.any(action[:-1] > 1) or np.any(action[:-1] < -1):
                    return None
                
            if step_idx == len(episode_raw_demo) - 1:
                reward = 1.0
                terminal = 1
                
            # if step_idx >= 20 and step_idx <=30: 
            #     print('prev_obs.joint_positions: ', episode_raw_demo[step_idx - 1].joint_positions)
            #     print('curr_obs.action: ', step_raw_demo.misc["joint_position_action"][:-1])
            #     print('curr_obs.joint_positions: ', step_raw_demo.joint_positions)
            #     print('accurcy: ', abs(step_raw_demo.joint_positions - step_raw_demo.misc["joint_position_action"][:-1]))
            #     print('action: ', action)
            #     print("---------------------------------------------------------------------------------")
                
            if self._config['custom_reward']:
                reward = reshape_reward_function(step_raw_demo, terminal, reward)
                
            episode_demo['observations'].append(obs)
            episode_demo['actions'].append(action)
            episode_demo['rewards'].append(reward)
            episode_demo['terminals'].append(terminal)
            
        return episode_demo

if __name__ == '__main__':
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
    print(desc)
    idx = 1
    while not done:
        tic = time.time()
        obs, reward, done, info = env.step(env.sample_action())
        # print(obs, reward, terminate, info)
        print(f"step {idx}: {time.time() - tic}")
        idx += 1
    print("=============================================")

    

    # demos = env.get_demos()
    # print(demos)