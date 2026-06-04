import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.ndimage import zoom
import nibabel as nib
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ==========================================
# 1. MOCK MODE CONFIGURATION & DATA LOADERS
# ==========================================
USE_MOCK = True   

if USE_MOCK:
    def dataloader(data_paths, batch_size): 
        idx = np.random.choice(len(data_paths), batch_size, replace=False)
        return [data_paths[i] for i in idx]

    def reptile_finetune_and_eval(selected_paths, val_paths): 
        # Simulates a validation Dice score after training on chosen images
        return float(np.random.uniform(0.4, 0.8))  
else:
    from dataset import dataloader
    from few_shot_val import reptile_finetune_and_eval

# ==========================================
# 2. CUSTOM 3D CNN POLICY NETWORK (THE BRAIN)
# ==========================================
class Custom3DCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # Expects shape input: (Batch_Size, Channels, Depth, Height, Width)
        # In our environment setup, this maps to: (32, 1, 32, 256, 256)
        self.cnn = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1), # Out: 16 x 16 x 128 x 128
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                # Out: 16 x 8 x 64 x 64
            
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),# Out: 32 x 4 x 32 x 32
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),                # Out: 32 x 2 x 16 x 16
            
            nn.Flatten()
        )
        
        # 32 channels * 2 slices * 16 height * 16 width = 16384 features
        self.linear = nn.Linear(32 * 2 * 16 * 16, features_dim)

    def forward(self, observations):
        # Flatten the outer batch/pool dimension to process all volumes through CNN
        batch_size, pool_size, c, d, h, w = observations.shape
        flat_obs = observations.view(batch_size * pool_size, c, d, h, w)
        
        features = self.linear(self.cnn(flat_obs))
        
        # Reshape back to the original batch configuration and mean across the pool
        features = features.view(batch_size, pool_size, -1)
        return torch.mean(features, dim=1)

# ==========================================
# 3. REINFORCEMENT LEARNING ENVIRONMENT
# ==========================================
class SimpleEnv(gym.Env):
    def __init__(self, data_paths, val_paths, batch_size=32, n_support=4, target_shape=(256, 256, 32)):
        super().__init__()

        self.data_paths = data_paths   
        self.val_paths = val_paths
        self.batch_size = batch_size   
        self.n_support = n_support
        self.target_shape = target_shape # (H, W, D) -> (256, 256, 32)

        # Transposing shape to match PyTorch requirements: (Channels, Depth, Height, Width)
        self.image_shape = (1, self.target_shape[2], self.target_shape[0], self.target_shape[1])
 
        # Observation Space shape: (Pool Size of 32 Candidate Images, 1, 32, 256, 256)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.batch_size, *self.image_shape), 
            dtype=np.float32
        )

        # Action Space: Output a raw selection probability score for each sample
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.batch_size,), 
            dtype=np.float32
        )
  
    def load_image(self, path):
        if USE_MOCK:
            # Generates a standard (H, W, D) mock image
            img = np.random.rand(*self.target_shape).astype(np.float32)   
        else:
            img = nib.load(path).get_fdata()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            factors = [t / s for t, s in zip(self.target_shape, img.shape)]
            img = zoom(img, factors, order=1) # Interpolates volume up/down to (256, 256, 32)

        # Rearrange matrix axes from (H, W, D) to (D, H, W) for PyTorch compatibility
        img = np.transpose(img, (2, 0, 1)) 
        return img[np.newaxis, ...] # Output shape: (1, 32, 256, 256)
    
    def select_obs(self):
        selected_paths = dataloader(self.data_paths, self.batch_size)
        self.current_paths = selected_paths

        images = [self.load_image(p) for p in selected_paths]
        obs = np.stack(images, axis=0).astype(np.float32)
        self.current_obs = obs  
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.select_obs()
        
        # Pick random support subset from current pool to establish baseline score
        random_support = np.random.choice(len(self.current_paths), self.n_support, replace=False)
        baseline_paths = [self.current_paths[i] for i in random_support]
        self.baseline_dice = reptile_finetune_and_eval(baseline_paths, self.val_paths)  
        
        return obs, {}

    def step(self, action):
        action = np.asarray(action).flatten()
        # Grab top-k index locations with highest scores outputted by PPO policy
        top_k = np.argsort(action)[-self.n_support:]                       
        selected_paths = [self.current_paths[i] for i in top_k]

        # Calculate step performance metric via our external optimization function
        dice = reptile_finetune_and_eval(selected_paths, self.val_paths)
        
        # UPDATED: Reward is now simply the raw validation dice score
        reward = dice

        # Move forward by loading fresh data pool
        obs = self.select_obs()

        terminated = False
        truncated = False
        info = {"dice": dice}

        return obs, reward, terminated, truncated, info

# ==========================================
# 4. TRAINING & EVALUATION PIPELINE
# ==========================================
def eval_agent(model, env, num_steps):
    obs, info = env.reset()
    total_dice = 0.0

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_dice += info["dice"]

    average_dice = total_dice / num_steps
    print(f"Average Dice over validation tracking phase: {average_dice:.4f}")
    return average_dice

def run_rl_active_selection(
    data_paths, val_paths, batch_size=32, n_support=4,
    target_shape=(256, 256, 32), train_trials=3, steps_per_trial=40, eval_steps=10
):
    env = SimpleEnv(
        data_paths=data_paths, val_paths=val_paths,
        batch_size=batch_size, n_support=n_support, target_shape=target_shape,
    )

    policy_kwargs = dict(features_extractor_class=Custom3DCNN)
    
    # Configured batch_size=2 and n_epochs=4 to protect RAM memory footprint bounds
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        verbose=1, 
        n_steps=steps_per_trial,
        batch_size=2,
        n_epochs=4
    )

    dice_history = []
    for trial in range(train_trials):
        model.learn(total_timesteps=steps_per_trial, reset_num_timesteps=False)
        print(f"\n--- Completed Trial Epoch {trial + 1}/{train_trials} ---")
        avg_dice = eval_agent(model, env, eval_steps)
        dice_history.append(avg_dice)

    return dice_history

# ==========================================
# 5. EXECUTION ENTRY POINT
# ==========================================
if __name__ == "__main__":
    fake_data_paths = [f"scan_case_{i}.nii.gz" for i in range(100)]   
    fake_val_paths  = [f"val_case_{i}.nii.gz" for i in range(10)]   

    print("Initiating active frame learning experiment script...")
    history = run_rl_active_selection(
        data_paths=fake_data_paths,
        val_paths=fake_val_paths,
        batch_size=32,
        n_support=4, 
        target_shape=(256, 256, 32),
        train_trials=3,
        steps_per_trial=40,
        eval_steps=5
    )
    print("\nTraining completed successfully! Performance profile tracking history:", history)