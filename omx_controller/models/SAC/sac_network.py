import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 256,  action_min=None, action_max=None):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)   
        self.log_std = nn.Linear(hidden_dim, action_dim)

        # Proper action scaling (device-safe)
        if action_min is not None and action_max is not None:
            action_scale = (action_max - action_min) / 2.0
            action_bias = (action_max + action_min) / 2.0
        else:
            action_scale = torch.ones(action_dim)
            action_bias = torch.zeros(action_dim)

        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state):
        x = self.backbone(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t) 


        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)

        mean_action = torch.tanh(mean)

        return y_t, log_prob, mean_action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.net(sa)
