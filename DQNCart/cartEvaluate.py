import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from DQN import DQN

import torch

env = gym.make("CartPole-v1", render_mode='human')
env._max_episode_steps = 10000  # Set to 1000 steps instead of 500

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

print(f"CUDA: {torch.cuda.is_available()}")
print(f"MPS: {torch.backends.mps.is_available()}")
print(f"device: {device}")

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load('policy_net_weights.pth'))

policy_net.eval()

target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

state, info = env.reset()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

# Loop for evaluating the model
for t in count():
    # Select action (without exploration since you're just evaluating)
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1, 1)

    # Take the action in the environment
    observation, reward, terminated, truncated, _ = env.step(action.item())

    # Move to the next state
    state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Check if the episode is done
    if terminated or truncated:
        print(f"Episode finished after {t + 1} steps")
        break
#  end evaluation only