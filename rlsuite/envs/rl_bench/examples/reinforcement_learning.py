import numpy as np
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import ReachTarget
from rlbench.observation_config import ObservationConfig

obs_config = ObservationConfig()
obs_config.set_all(False)
obs_config.set_all_high_dim(False)
obs_config.set_all_low_dim(False)

action_mode = MoveArmThenGripper(
  arm_action_mode=JointVelocity(),
  gripper_action_mode=Discrete()
)
env = Environment(action_mode, 
                  obs_config=obs_config,
                  headless=True)
env.launch()

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()


import time

for _ in range(100):
  
  tic = time.time()
  obs, reward, terminate = task.step(np.random.normal(size=env.action_shape))
  print(f"time per step: {time.time() - tic}")
  # print(f"obs: {obs.__dict__.keys()}")
  # print(f"reward: {reward}")
  # print(f"terminate: {terminate}")
