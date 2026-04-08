# import copy
# import torch
# from omx_controller.models.IQL.network import QNetwork, VNetwork, Policy
# class IQLAgent:
#     def __init__(self, state_dim, action_dim,
#                  tau=0.7, beta=3.0, gamma=0.99, ema=0.005):

#         # Double Q
#         self.q1 = QNetwork(state_dim, action_dim)
#         self.q2 = QNetwork(state_dim, action_dim)

#         self.v = VNetwork(state_dim)
#         self.v_target = copy.deepcopy(self.v)  # target V

#         self.policy = Policy(state_dim, action_dim)

#         self.q_opt = torch.optim.Adam(
#             list(self.q1.parameters()) + list(self.q2.parameters()), 3e-4)
#         self.v_opt = torch.optim.Adam(self.v.parameters(), 3e-4)
#         self.pi_opt = torch.optim.Adam(self.policy.parameters(), 3e-4)

#         self.tau = tau
#         self.beta = beta
#         self.gamma = gamma
#         self.ema = ema
# import copy
# import torch
# import torch.nn.functional as F
# from omx_controller.models.IQL.network import QNetwork, ValueNetwork, Actor

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# class IQLAgent:
#     def __init__(self, state_dim, action_dim):

#         self.q1 = QNetwork(state_dim, action_dim).to(DEVICE)
#         self.q2 = QNetwork(state_dim, action_dim).to(DEVICE)
#         self.q_target = copy.deepcopy(self.q1)

#         self.v = ValueNetwork(state_dim).to(DEVICE)
#         self.actor = Actor(state_dim, action_dim).to(DEVICE)

#         self.q_optimizer = torch.optim.Adam(
#             list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
#         )
#         self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=3e-4)
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

#         self.discount = 0.99
#         self.tau = 0.005

#         self.expectile = 0.7
#         self.beta = 3.0

#     def train(self, batch):
#         s, a, r, s_next, done = batch

#         s = s.to(DEVICE)
#         a = a.to(DEVICE)
#         r = r.to(DEVICE)
#         s_next = s_next.to(DEVICE)
#         done = done.to(DEVICE)

#         # ---------------------
#         # 1. Update V
#         # ---------------------
#         with torch.no_grad():
#             q1 = self.q1(s, a)
#             q2 = self.q2(s, a)
#             q = torch.min(q1, q2)

#         v = self.v(s)
#         v_loss = expectile_loss(q - v, self.expectile).mean()

#         self.v_optimizer.zero_grad()
#         v_loss.backward()
#         self.v_optimizer.step()

#         # ---------------------
#         # 2. Update Q
#         # ---------------------
#         with torch.no_grad():
#             target_v = self.v(s_next)
#             target_q = r + (1 - done) * self.discount * target_v

#         q1 = self.q1(s, a)
#         q2 = self.q2(s, a)

#         q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

#         self.q_optimizer.zero_grad()
#         q_loss.backward()
#         self.q_optimizer.step()

#         # ---------------------
#         # 3. Update Actor
#         # ---------------------
#         with torch.no_grad():
#             v = self.v(s)
#             q = torch.min(self.q1(s, a), self.q2(s, a))
#             advantage = q - v
#             weight = torch.exp(advantage / self.beta).clamp(max=100.0)

#         log_prob = self.actor.log_prob(s, a)
#         actor_loss = -(weight * log_prob).mean()

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         # ---------------------
#         # 4. Soft update
#         # ---------------------
#         for param, target_param in zip(self.q1.parameters(), self.q_target.parameters()):
#             target_param.data.copy_(
#                 self.tau * param.data + (1 - self.tau) * target_param.data
#             )

#         return {
#             "v_loss": v_loss.item(),
#             "q_loss": q_loss.item(),
#             "actor_loss": actor_loss.item()
#         }
    
# def expectile_loss(diff, tau):
#     weight = torch.where(diff > 0, tau, 1 - tau)
#     return weight * diff.pow(2)
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from omx_controller.models.IQL.network import MLP, ValueNet, Policy
import pandas as pd
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class IQL:
    def __init__(self, state_dim, action_dim, device):

        self.q = MLP(state_dim, action_dim).to(device)
        self.q_target = copy.deepcopy(self.q)

        self.v = ValueNet(state_dim).to(device)
        self.pi = Policy(state_dim, action_dim).to(device)

        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=3e-4)
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=3e-4)
        self.pi_opt = torch.optim.Adam(self.pi.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.tau = 0.7          # expectile
        self.beta = 3.0         # advantage temperature
        self.device = device

    # =========================
    # Expectile loss
    # =========================
    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self.tau, 1 - self.tau)
        return weight * diff.pow(2)

    # =========================
    # Training step
    # =========================
    def update(self, batch):
        s, a, r, s_next, done = batch

        # =====================
        # 1. Update Value V
        # =====================
        with torch.no_grad():
            q_val = self.q_target(s, a)

        v = self.v(s)
        diff = q_val - v
        v_loss = self.expectile_loss(diff).mean()

        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()

        # =====================
        # 2. Update Q
        # =====================
        with torch.no_grad():
            target_v = self.v(s_next)
            target_q = r + self.gamma * (1 - done) * target_v

        q = self.q(s, a)
        q_loss = F.mse_loss(q, target_q)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # =====================
        # 3. Update Policy (AWR)
        # =====================
        with torch.no_grad():
            adv = self.q(s, a) - self.v(s)
            weights = torch.exp(self.beta * adv).clamp(max=100.0)

        log_prob = self.pi.log_prob(s, a)
        pi_loss = -(weights * log_prob).mean()

        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()

        # =====================
        # 4. Update target Q
        # =====================
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "pi_loss": pi_loss.item()
        }
    
    def save_checkpoint(self, path):
            # Đảm bảo thư mục tồn tại
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            checkpoint = {
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "v": self.v.state_dict(),
                "pi": self.pi.state_dict(),
                "q_opt": self.q_opt.state_dict(),
                "v_opt": self.v_opt.state_dict(),
                "pi_opt": self.pi_opt.state_dict(),
            }
            torch.save(checkpoint, path)
            print(f"[✓] Đã lưu IQL checkpoint tại: {path}")

    def load_checkpoint(self, path):
            checkpoint = torch.load(path, map_location=self.device)
            
            self.q.load_state_dict(checkpoint["q"])
            self.q_target.load_state_dict(checkpoint["q_target"])
            self.v.load_state_dict(checkpoint["v"])
            self.pi.load_state_dict(checkpoint["pi"])
            
            self.q_opt.load_state_dict(checkpoint["q_opt"])
            self.v_opt.load_state_dict(checkpoint["v_opt"])
            self.pi_opt.load_state_dict(checkpoint["pi_opt"])
            
            print(f"[✓] Đã load IQL checkpoint từ: {path}")

def train_iql(iql, dataset, epochs=100, batch_size=256, log_interval=10, save_dir="omx_controller/models/IQL/checkpoint"):
    os.makedirs(save_dir, exist_ok=True)

    dataset = dataset.with_format("torch")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # 🔥 tránh lỗi multiprocess
        pin_memory=True,
        drop_last=True
    )

    for epoch in range(epochs):
        epoch_losses = {"v_loss": 0, "q_loss": 0, "pi_loss": 0}

        for batch in dataloader:
            # =========================
            # 🔥 build state (9D)
            # =========================
            s = torch.stack([
                batch["s1"], batch["s2"], batch["s3"],
                batch["s4"], batch["s5"], batch["g_s"],
                batch["rb_x"], batch["rb_y"], batch["rb_z"]
            ], dim=1).to(iql.device)

            # =========================
            # 🔥 action (6D)
            # =========================
            a = torch.stack([
                batch["a1"], batch["a2"], batch["a3"],
                batch["a4"], batch["a5"], batch["g_a"]
            ], dim=1).to(iql.device)

            # =========================
            # 🔥 next state (9D)
            # =========================
            s_next = torch.stack([
                batch["ns1"], batch["ns2"], batch["ns3"],
                batch["ns4"], batch["ns5"], batch["ng_s"],
                batch["nrb_x"], batch["nrb_y"], batch["nrb_z"]
            ], dim=1).to(iql.device)

            # =========================
            # reward & done
            # =========================
            r = batch["reward"].unsqueeze(-1).to(iql.device)
            done = batch["done"].float().unsqueeze(-1).to(iql.device)

            batch_torch = (s, a, r, s_next, done)

            losses = iql.update(batch_torch)

            for k, v in losses.items():
                epoch_losses[k] += v

        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"V: {epoch_losses['v_loss']/len(dataloader):.6f} | "
                  f"Q: {epoch_losses['q_loss']/len(dataloader):.6f} | "
                  f"Pi: {epoch_losses['pi_loss']/len(dataloader):.6f}")

            iql.save_checkpoint(f"{save_dir}/iql_epoch_{epoch+1}.pth")

    iql.save_checkpoint(f"{save_dir}/iql_final.pth")
    print("=== IQL Training Finished ===")

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_files = [
        os.path.join("omx_controller/models/SAC/logs_1", f"log_{i}.csv")
        for i in range(1, 21)
    ]

    dataset = load_dataset("csv", data_files=data_files, split="train")

    # 🔥 fixed theo dataset của bạn
    state_dim = 9
    action_dim = 6

    print("State dim:", state_dim)
    print("Action dim:", action_dim)
    print("Dataset size:", len(dataset))

    iql = IQL(state_dim, action_dim, DEVICE)

    train_iql(iql, dataset, epochs=200, batch_size=512, log_interval=5)