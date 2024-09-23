from gymnasium.envs.registration import register

register(
     id="ConnectFour-v0",
     entry_point="connect_four_gym.envs.connectfour:ConnectFourEnv",
     max_episode_steps=300,
)