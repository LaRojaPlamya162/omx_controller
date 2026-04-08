# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# LOG_STD_MIN = -5
# LOG_STD_MAX = 2


# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, s, a):
#         return self.net(torch.cat([s, a], dim=-1))


# class VNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, s):
#         return self.net(s)


# class Policy(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU()
#         )
#         self.mean = nn.Linear(256, action_dim)
#         self.log_std = nn.Linear(256, action_dim)

#     def forward(self, s):
#         h = self.net(s)
#         mean = self.mean(h)
#         log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
#         return mean, log_std

#     def sample(self, s):
#         mean, log_std = self(s)
#         std = log_std.exp()
#         dist = torch.distributions.Normal(mean, std)

#         a = dist.rsample()  # FIX
#         log_prob = dist.log_prob(a).sum(-1)  # FIX

#         return a, log_prob

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, s, a):
#         x = torch.cat([s, a], dim=-1)
#         return self.net(x)
    
# class ValueNetwork(nn.Module):
#     def __init__(self, state_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, s):
#         return self.net(s)
    
# LOG_STD_MIN = -5
# LOG_STD_MAX = 2

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU()
#         )
#         self.mu = nn.Linear(256, action_dim)
#         self.log_std = nn.Linear(256, action_dim)

#     def forward(self, s):
#         x = self.net(s)
#         mu = self.mu(x)
#         log_std = self.log_std(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
#         std = log_std.exp()
#         return mu, std

#     def log_prob(self, s, a):
#         mu, std = self.forward(s)
#         dist = torch.distributions.Normal(mu, std)
#         return dist.log_prob(a).sum(-1, keepdim=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset
# =========================
# Networks
# =========================
class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


class ValueNet(nn.Module):
    def __init__(self, state_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, s):
        return self.net(s)


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = self.net(s)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        std = torch.exp(log_std)
        return mean, std

    def log_prob(self, s, a):
        mean, std = self(s)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(a).sum(-1, keepdim=True)

    def act(self, s, deterministic=True):
        with torch.no_grad():
            mean, std = self(s)

            if deterministic:
                action = mean
            else:
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()

            return action
