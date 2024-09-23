import torch
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

def train():
    # Create environment
    env = gym.make("Humanoid")
    log_dir = "./ppo_cartpole_tensorboard/"

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create PPO model
    model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir)
    # model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir)
    # Train the model
    model.learn(total_timesteps=1000000)

    # Save the model
    model.save("ppo_cartpole")
    print("Done")


def run():
    env = gym.make("Humanoid", render_mode='human')
    model = TD3.load("ppo_cartpole")
    # model = PPO.load("ppo_cartpole")

    # Test the model
    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

train()