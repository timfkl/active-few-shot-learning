import gymnasium as gym
from gymnasium import spaces
import numpy as np

from stable_baselines3 import PPO

from data_loader import build_nifti_batch_generator
from MOCK_few_shot_val import reptile_finetune_and_eval


# A sample batch is loaded before the main loop to confirm the image shape
# Assum the image is resized and standardized,so 
# self.observation_space = spaces.Box(
#             low=-10, high=10,
#             shape=(self.batch_size, *self.image_shape),
#             dtype=np.float32,
# )

# Assum reptile_finetune_and_eval trains on the selected image and lables, returns the Dice score
# reward=dice
#The data fed into reptile_finetune_and_eval is 4-dimensional tensor

#use (16, 16, 4) mock data, both train and val 




# Environment setup
class SimpleEnv(gym.Env):
    def __init__(self,
                  val_paths="data-resize-val",
                  images_dir="data-resize", 
                  masks_dir="data-resize",
                  batch_size=32,n_support=1):
        super().__init__()

        self.val_paths=val_paths
        self.batch_size = batch_size   
        self.n_support = n_support


        self.gen = build_nifti_batch_generator(
            batch_size=self.batch_size,
            images_dir=images_dir,
            masks_dir=masks_dir,
            normalize=True,
        )
        
        # get one bacth to identify the real shape
        sample_imgs, sample_lbls = next(self.gen)
        self.image_shape = sample_imgs.shape[1:]          

        self.observation_space = spaces.Box(
            # since the image is standardized
            low=-10, high=10,
            shape=(self.batch_size, *self.image_shape),
            dtype=np.float32,
        )

        # Action space: output a score for each of the 32 candidate samples
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

        dice = reptile_finetune_and_eval(selected_images, selected_labels, self.val_paths)
        reward = dice 

        obs = self.select_obs()

        terminated = True
        truncated = False
        info = {"dice": dice}

        return obs, reward, terminated, truncated, info


# Evaluate the agent after several rounds of training
def eval_agent(model, env, num_steps):
    obs, info = env.reset()
    total_dice = 0.0

    for _ in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_dice += info["dice"]

    average_dice = total_dice / num_steps
    print(f"Average dice over {num_steps} steps: {average_dice:.4f}")
    return average_dice


# Create the sample-selection environment, train a PPO model on it
# evaluate the agent after each training trial, and return the Dice history.
def run_rl_active_selection(
    val_paths,
    batch_size=32,
    n_support=1,
    train_trials=10,
    steps_per_trial=256,
    eval_steps=100,
):
    env = SimpleEnv(
        val_paths=val_paths,
        batch_size=batch_size,
        n_support=n_support,
    )

    model = PPO("MlpPolicy", env, n_steps=steps_per_trial, batch_size=32,verbose=0)

    dice_history = []

    for trial in range(train_trials):
        model.learn(total_timesteps=steps_per_trial)
        print(f"Trial {trial + 1}/{train_trials}")
        avg_dice = eval_agent(model, env, eval_steps)
        dice_history.append(avg_dice)

    return dice_history  # One Dice value for each trial, for example: [0.42, 0.55, ...]

#Test the RL environment with several random actions
if __name__ == "__main__":
    dice_history = run_rl_active_selection(
        val_paths="data-resize-val",
        train_trials=2,       
        steps_per_trial=32,
        eval_steps=10,
    )
    print("dice history:", dice_history)