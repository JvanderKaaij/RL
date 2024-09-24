import gymnasium as gym
from torch.optim import AdamW
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage, MultiStep, LazyTensorStorage
from torchrl.envs import GymEnv
from torchrl.modules import ConvNet, QValueActor
from torchrl.objectives import DQNLoss
from torchrl.data import OneHotDiscreteTensorSpec
from torchrl.envs import GymWrapper, step_mdp, TransformedEnv, Compose, ObservationNorm
import connect_four_gym
from stable_baselines3.common.buffers import ReplayBuffer

from torchrl.envs import (
    EnvCreator,
    ExplorationType,
    ParallelEnv,
    RewardScaling,
    StepCounter,
)


def train():
    log_dir = "./connectfour-log/"
    gamma = 0.99
    LR = 1e-4

    policy = ConvNet(in_features=3, depth=1, num_cells=[32, ])
    target = ConvNet(in_features=3, depth=1, num_cells=[32, ])
    buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=5)

    base_env = gym.make('ConnectFour-v0')
    env = GymWrapper(base_env)
    env = TransformedEnv(env, ObservationNorm(in_keys=["observation"]))

    action_spec = env.action_spec
    qvalue_actor = QValueActor(policy, spec=action_spec, in_keys=["observation"])

    reset = env.reset()
    print(reset)

    action_dict = qvalue_actor(reset)

    reset_with_action = env.rand_action(reset)

    stepped_data = env.step(action_dict)

    data = step_mdp(stepped_data)

    dqn_loss = DQNLoss(policy, action_space=qvalue_actor.action_space)

    optimizer = AdamW(policy.parameters(), lr=LR, amsgrad=True)


train()
