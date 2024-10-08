import torch
from torch import nn
from torch.nn import functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        activation1 = F.relu(self.layer1(obs_tensor))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output
