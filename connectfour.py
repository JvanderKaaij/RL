import gymnasium as gym
from torchrl.collectors import SyncDataCollector, MultiaSyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, MultiStep, LazyTensorStorage
from torchrl.envs import GymEnv
from torchrl.modules import MLP, ConvNet

import connect_four_gym
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)


def train():
    env = GymEnv('ConnectFour-v0')
    reset = env.reset()
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=5)
    model = ConvNet(in_features=3, depth=1, num_cells=[32,])

    print(reset)
    log_dir = "./connectfour-log/"


train()
