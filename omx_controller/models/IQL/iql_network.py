import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0

# =========================
# Networks
# =========================
class QNetwork(nn.Module):
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
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def log_prob(self, s, a_scaled):
        """
        a_scaled: action từ dataset đã được scale về [-1, 1]
        """
        mean, log_std = self.forward(s)
        std = log_std.exp()

        # a_scaled đã ở [-1, 1] → chuyển về raw space để tính Gaussian
        a_raw = torch.atanh(torch.clamp(a_scaled, -0.9999, 0.9999))

        dist = torch.distributions.Normal(mean, std)
        log_p = dist.log_prob(a_raw)

        # Jacobian correction cho tanh
        log_p -= torch.log(1 - a_scaled.pow(2) + 1e-6)

        return log_p.sum(-1, keepdim=True)

    def act(self, s, deterministic=True):
        """Trả về action đã scale về [-1, 1]"""
        with torch.no_grad():
            mean, log_std = self.forward(s)
            if deterministic:
                a_raw = mean
            else:
                std = log_std.exp()
                a_raw = torch.distributions.Normal(mean, std).sample()

            a_scaled = torch.tanh(a_raw)
            return a_scaled   # ← Đây là dạng bạn muốn (scaled [-1,1])