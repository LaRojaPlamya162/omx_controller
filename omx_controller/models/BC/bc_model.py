# import numpy as np
# from datasets import load_dataset
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import torch.distributions as D
# import os

# LOG_STD_MIN = -5
# LOG_STD_MAX = 2
# class BCPolicy(nn.Module):
#     def __init__(self, state_dim, action_dim, hidden_dim=128):
#         super().__init__()

#         self.backbone = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#         )

#         # Level 1: output parameters of a distribution
#         self.mean = nn.Linear(hidden_dim, action_dim)
#         self.log_std = nn.Linear(hidden_dim, action_dim)

#     def forward(self, state):
#         """
#         Return a distribution π(a|s)
#         """
#         h = self.backbone(state)
#         mean = self.mean(h)
#         log_std = self.log_std(h)
#         log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
#         std = torch.exp(log_std)

#         return D.Normal(mean, std)

#     def act(self, state, deterministic=False):
#         """
#         Sample or take mean action (for deployment)
#         """
#         dist = self.forward(state)
#         if deterministic:
#             return dist.mean
#         else:
#             return dist.sample()

# def cut_back_attribute(example, col_name, n):
#         example[col_name] = example[col_name][:n]
#         return example 
# def bc_nll_loss(dist, actions):
#     log_prob = dist.log_prob(actions)
#     return -log_prob.sum(dim=-1).mean()
# if __name__ == '__main__':
#     BASE_PATH = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/LearnDLFromScratch/omx_f_PickUp/data/chunk-000"

#     data_files = [
#     os.path.join(BASE_PATH, f"episode_{i:06d}.parquet")
#     for i in range(15)
# ]

#     dataset = load_dataset("parquet", data_files=data_files,split="train")
#     #dataset = load_dataset("RobotisSW/omx_Move", split = "train")
#     dataset = dataset.with_format("torch")
#     #dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "action", "n" : 6})
#     #dataset = dataset.map(cut_back_attribute, fn_kwargs = {"col_name" : "observation.state", "n" : 6})
#     dataloader = DataLoader(
#         dataset = dataset,
#         batch_size = 256,
#         shuffle = True,
#         num_workers = 2,
#         pin_memory = True,
#         drop_last = True
#     )
#     model = BCPolicy(state_dim = len(dataset[0]['observation.state']),
#                      action_dim = len(dataset[0]['action']))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

#     model.train()
#     for epoch in range(50):
#         total_loss = 0
#         for batch in dataloader:
#         #for i in range(len(dataset)):
#             state = batch['observation.state']
#             action = batch['action']

#             dist = model(state)
#             loss = bc_nll_loss(dist, action)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(dataloader):.6f}")
    
#     print("Training finised")
#     torch.save(model.state_dict(), 'omx_controller/models/BC/bc_model.pth')
#     #torch.save(model.state_dict(), 'src/omx_controller/omx_controller/models/BC/real_bc_model.pth')
#     print("Model loaded!")

import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

LOG_STD_MIN = -5
LOG_STD_MAX = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.backbone(state)

        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        return torch.distributions.Normal(mean, std)

    def act(self, state, deterministic=True):
        dist = self.forward(state)

        action = dist.mean if deterministic else dist.sample()

        # squash để giới hạn [-1,1]
        action = torch.tanh(action)

        return action
    
def compute_stats(dataset):

    states = torch.stack([dataset[i]["observation.state"] for i in range(len(dataset))])
    actions = torch.stack([dataset[i]["action"] for i in range(len(dataset))])

    state_mean = states.mean(0)
    state_std = states.std(0) + 1e-6

    action_mean = actions.mean(0)
    action_std = actions.std(0) + 1e-6

    return state_mean, state_std, action_mean, action_std

def load_model(path, state_dim, action_dim):
    checkpoint = torch.load(path, map_location=DEVICE)

    model = BCPolicy(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint

if __name__ == '__main__':

    BASE_PATH = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/LearnDLFromScratch/omx_f_PickUp/data/chunk-000"

    data_files = [
        os.path.join(BASE_PATH, f"episode_{i:06d}.parquet")
        for i in range(15)
    ]

    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = dataset.with_format("torch")

    state_dim = len(dataset[0]["observation.state"])
    action_dim = len(dataset[0]["action"])

    state_mean, state_std, action_mean, action_std = compute_stats(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    model = BCPolicy(state_dim, action_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(50):
        total_loss = 0

        for batch in dataloader:
            state = batch["observation.state"].to(DEVICE)
            action = batch["action"].to(DEVICE)

            # normalize
            state = (state - state_mean.to(DEVICE)) / state_std.to(DEVICE)
            action = (action - action_mean.to(DEVICE)) / action_std.to(DEVICE)

            dist = model(state)

            log_prob = dist.log_prob(action)
            loss = -log_prob.sum(dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(dataloader):.6f}")

    print("Training finished")

    os.makedirs("models", exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
    }, "omx_controller/models/BC/real_bc_model.pth")