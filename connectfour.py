import gymnasium as gym
from stable_baselines3 import DQN, PPO
import connect_four_gym


def train():
    env = gym.make('ConnectFour-v0')
    log_dir = "./connectfour-log/"
    model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    model.learn(1000)
    model.save("dqn_connectfour")


train()
