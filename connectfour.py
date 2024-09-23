import gymnasium as gym
from stable_baselines3 import DQN, PPO
import connect_four_gym
from stable_baselines3.common.buffers import ReplayBuffer

def train():
    env = gym.make('ConnectFour-v0')
    log_dir = "./connectfour-log/"

    # TODO: Alternate an x amount of times:
    agent_one_model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    agent_one_model.learn(10000)
    # obs, info = env.reset()
    # while True:
    #     action, _states = agent_one_model.predict(obs)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #
    #     if terminated or truncated:
    #         obs, info = env.reset()


    # agent_two_model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_dir)
    # agent_two_model.learn(10000)



train()
