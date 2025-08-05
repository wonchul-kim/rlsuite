        
import gymnasium as gym 
import numpy as np 
import torch 

class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    
if __name__ == '__main__':
    import random 
    from agent import BCAgent
    
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = ActionNormalizer(env)

    seed = 777
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    
    import pickle

    # load demo on replay memory
    demo_path = "/HDD/etc/pg-is-all-you-need-master/demo.pkl"
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
        
    # parameters
    num_frames = 50000
    memory_size = 100000
    batch_size = 1024
    demo_batch_size = 128
    ou_noise_theta = 1.0
    ou_noise_sigma = 0.1
    initial_random_steps = 10000

    agent = BCAgent(
        env,
        memory_size,
        batch_size,
        demo_batch_size,
        ou_noise_theta,
        ou_noise_sigma,
        demo,
        initial_random_steps=initial_random_steps,
        seed=seed,
    )
    
    agent.train(num_frames, output_dir='/HDD/etc/rl/bc')
    video_folder = "/HDD/etc/rl/bc"
    agent.test(video_folder=video_folder)