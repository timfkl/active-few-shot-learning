"""
RL active support selection for few-shot 3D segmentation.

A PPO agent selects support samples from a candidate batch; the reward is the
query-set Dice of a Reptile 3D U-Net fine-tuned on the selected samples.
Flow: data_loader -> SimpleEnv (gym) -> PPO -> reptile. Run: python RL_second_draft.py
Requires a pretrained Reptile init at WEIGHTS_PATH ('reptile_init.pt').
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

from stable_baselines3 import PPO
from tqdm import tqdm

from data_loader import build_nifti_batch_generator

import torch                                    
from reptile import UNet3D, adapt_on_support, evaluate_on_query   

# Assume the file 'reptile_init.pt' already exists somewhere; raise an error if it does not.
WEIGHTS_PATH = "reptile_init.pt"
_INIT_STATE = None  # Cache the state_dict to avoid reading from disk every time it is called.

def _get_init_state(weights_path=WEIGHTS_PATH):
    global _INIT_STATE

def _get_init_state(weights_path=WEIGHTS_PATH):
    global _INIT_STATE
    if _INIT_STATE is None:
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Reptile init weights '{weights_path}' not found. "
                f"Please run train_reptile and save the weights first, "
                f"or place the weights file at this path."
            )
        _INIT_STATE = torch.load(weights_path, map_location="cpu")
    return _INIT_STATE

def reptile_fine_tune_eval(support_images, support_masks, query_images, query_masks):
    model = UNet3D()
    model.load_state_dict(_get_init_state())  
    adapted_model = adapt_on_support(model, support_images, support_masks)
    result = evaluate_on_query(adapted_model, query_images, query_masks)
    return result["dice"]

# A sample batch is loaded before the main loop to confirm the image shape and used to be the fixed query set


# Environment setup
class SimpleEnv(gym.Env):
    def __init__(self,
                  images_dir="data-resize", 
                  masks_dir="data-resize",
                  batch_size=16,
                  n_support=1,
                  task_number=1):
        super().__init__()

        self.batch_size = batch_size   
        self.n_support = n_support
        self.task_number = task_number


        self.gen = build_nifti_batch_generator(
            batch_size=self.batch_size,
            images_dir=images_dir,
            masks_dir=masks_dir,
            normalize=True,
            task_number=self.task_number
        )
        
        # get one bacth to identify the real shape, and fix this bactch to be query set
        sample_imgs, sample_lbls = next(self.gen)
        self.image_shape = sample_imgs.shape[1:]          

        if sample_lbls.ndim == sample_imgs.ndim - 1:
            sample_lbls = sample_lbls[:, None, ...]
        self.query_images = sample_imgs               
        self.query_masks = sample_lbls 

        self.observation_space = spaces.Box(
            # since the image is standardized
            low=0.0, 
            high=1.0,
            shape=(self.batch_size, *self.image_shape),
            dtype=np.float32,
        )

        # Action space: output a score for each of the 16 candidate samples
        self.action_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.batch_size,), 
            dtype=np.float32
        )
    
    # Select batch_size images from all paths and convert them into a 4D tensor
    def select_obs(self):
        batch_images, batch_labels = next(self.gen)
        #  The mask has shape `(B, H, W, D)` without a channel dimension, so here we add one to make it `(B, 1, H, W, D)`.
        if batch_labels.ndim == batch_images.ndim - 1:
            batch_labels = batch_labels[:, None, ...]
        self.current_images = batch_images
        self.current_labels = batch_labels
        self.current_obs = batch_images
        return batch_images

    # Reset the environment, generate a new observation batch
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.select_obs()
        info = {}
        return obs, info

    # Select the top-scoring support samples based on action,
    # evaluate them on the fixed validation set, calculate the reward, 
    # return a new observation batch.
    def step(self, action):
        action = np.asarray(action).flatten()
        top_k = np.argsort(action)[-self.n_support:]                       

        selected_images = self.current_images[top_k]
        selected_labels = self.current_labels[top_k]

        dice = reptile_fine_tune_eval(selected_images, selected_labels, self.query_images, self.query_masks,)
        reward = dice 

        obs = self.current_obs

        terminated = True
        truncated = False
        info = {"dice": dice}

        return obs, reward, terminated, truncated, info


# Evaluate the agent after several rounds of training
def eval_agent(model, env, num_steps):
    total_dice = 0.0
    obs, info = env.reset()
    for _ in tqdm(range(num_steps), desc="RL evaluation", leave=False, dynamic_ncols=True):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_dice += info["dice"]
        if terminated or truncated:
            obs, info = env.reset()
    average_dice = total_dice / num_steps
    return average_dice

# Create the sample-selection environment, train a PPO model on it
# evaluate the agent after each training trial, and return the Dice history.
def run_rl_active_selection(
    batch_size=16,
    n_support=4,
    task_number=1,
    train_trials=10,
    steps_per_trial=256,
    eval_steps=100,
):
    print("[setup] building environment ...")                                        
    env = SimpleEnv(
        batch_size=batch_size,
        n_support=n_support,
        task_number=task_number
    )
    print("[setup] environment ready, creating PPO model ...")                       

    model = PPO("MlpPolicy", env, n_steps=steps_per_trial, batch_size=32, verbose=0)
    print("[setup] model ready, start training\n")                                   

    dice_history = []

    trial_bar = tqdm(range(train_trials), desc="RL trials", dynamic_ncols=True)
    for trial in trial_bar:
        model.learn(total_timesteps=steps_per_trial)
        avg_dice = eval_agent(model, env, eval_steps)
        dice_history.append(avg_dice)
        trial_bar.set_postfix(avg_dice=f"{avg_dice:.4f}")

    return dice_history


if __name__ == "__main__":
    print(">>> run_rl_active_selection starting")                                    
    dice_history = run_rl_active_selection(
        train_trials=2,
        steps_per_trial=32,
        eval_steps=10,
    )
    print(">>> done. dice history:", dice_history)
