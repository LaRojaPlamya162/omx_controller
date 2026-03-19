from omx_controller.models.SAC.network import Actor, Critic
import torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_STD_MIN = -20
LOG_STD_MAX = 2
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.q1 = Critic(state_dim, action_dim)
        self.q2 = Critic(state_dim, action_dim)

        self.q1_target = Critic(state_dim, action_dim)
        self.q2_target = Critic(state_dim, action_dim)

        self.actor = self.actor.to(DEVICE)
        self.q1 = self.q1.to(DEVICE)
        self.q2 = self.q2.to(DEVICE)
        self.q1_target = self.q1_target.to(DEVICE)
        self.q2_target = self.q2_target.to(DEVICE)
    

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=3e-4)
        
        # ===== SAC v2: auto entropy =====
        self.target_entropy = -action_dim
        #self.log_alpha = torch.zeros(1, requires_grad = True)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.gamma = 0.99
        #self.alpha = 0.2
        self.tau = 0.005

    def update(self, replay, batch_size=256):

        # ReplayBuffer đã trả về tensor trên đúng device
        s, a, r, s_, d = replay.sample(batch_size)

        alpha = self.log_alpha.exp()

        # =========================
        # Critic update
        # =========================
        with torch.no_grad():

            a_next, logp_next, _ = self.actor.sample(s_)

            q1_target = self.q1_target(s_, a_next)
            q2_target = self.q2_target(s_, a_next)

            min_q_target = torch.min(q1_target, q2_target)

            target_q = r + self.gamma * (1 - d) * (min_q_target - alpha * logp_next)

        # current Q estimates
        q1_current = self.q1(s, a)
        q2_current = self.q2(s, a)

        q1_loss = F.mse_loss(q1_current, target_q)
        q2_loss = F.mse_loss(q2_current, target_q)

        self.q1_opt.zero_grad()
        q1_loss.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        q2_loss.backward()
        self.q2_opt.step()

        # =========================
        # Actor update
        # =========================
        a_new, logp, _ = self.actor.sample(s)

        q1_new = self.q1(s, a_new)
        q2_new = self.q2(s, a_new)

        q_new = torch.min(q1_new, q2_new)

        actor_loss = (alpha * logp - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # =========================
        # Alpha update (SAC v2)
        # =========================
        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # =========================
        # Soft update target networks
        # =========================
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save_checkpoint(self, path):
            torch.save({
                "actor": self.actor.state_dict(),
                "q1": self.q1.state_dict(),
                "q2": self.q2.state_dict(),
                "q1_target": self.q1_target.state_dict(),
                "q2_target": self.q2_target.state_dict(),
                "actor_opt": self.actor_opt.state_dict(),
                "q1_opt": self.q1_opt.state_dict(),
                "q2_opt": self.q2_opt.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "alpha_opt": self.alpha_opt.state_dict(),
            }, path)
            print(f"[✓] Saved to {path}")

    
    def load_checkpoint(self, path):
        #ckpt = torch.load(path, map_location=self.device)
        #ckpt = torch.load(path)
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])

        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])

        #self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.log_alpha.data.copy_(ckpt['log_alpha'].to(DEVICE))
        #self.log_alpha.data.copy_(ckpt["log_alpha"])
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        print(f"[✓] Loaded from {path}")




# import torch
# import torch.nn.functional as F
# from omx_controller.models.SAC.network import Actor, Critic

# class SACAgent:
#     def __init__(self, state_dim, action_dim, device = "cpu"):
#         self.device = device

#         # === Networks ===
        

#         self.actor = Actor(state_dim, action_dim).to(device)

#         self.critic = Critic(state_dim, action_dim).to(device)
#         self.critic_target = Critic(state_dim, action_dim).to(device)
#         self.critic_target.load_state_dict(self.critic.state_dict())

#         # === Optimizers ===
#         self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
#         self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

#         # === Entropy Temperature (Auto α tuning) ===
#         self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
#         self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

#         self.alpha = self.log_alpha.exp().detach()
#         self.target_entropy = -action_dim  # SAC paper heuristic

#         # === Hyperparameters ===
#         self.gamma = 0.99
#         self.tau = 0.005

#     # ============================================================
#     # Select Action (interaction with env / robot)
#     # ============================================================
#     def select_action(self, state, evaluate=False):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             if evaluate:
#                 _, _, action = self.actor.sample(state)  # deterministic
#             else:
#                 action, _, _ = self.actor.sample(state)

#         return action.cpu().numpy()[0]

#     # ============================================================
#     # SAC Update Step
#     # ============================================================
#     def update(self, replay_buffer, batch_size=256):
#         s, a, r, s_, d = replay_buffer.sample(batch_size)

#         # -----------------------------
#         # Compute Target Q
#         # -----------------------------
#         with torch.no_grad():
#             a_next, logp_next, _ = self.actor.sample(s_)

#             q1_t, q2_t = self.critic_target(s_, a_next)
#             q_target = r + self.gamma * (1 - d) * (
#                 torch.min(q1_t, q2_t) - self.alpha * logp_next
#             )

#         # -----------------------------
#         # Critic Update
#         # -----------------------------
#         q1, q2 = self.critic(s, a)

#         critic_loss = (
#             F.mse_loss(q1, q_target) +
#             F.mse_loss(q2, q_target)
#         )

#         self.critic_opt.zero_grad()
#         critic_loss.backward()
#         self.critic_opt.step()

#         # -----------------------------
#         # Actor Update
#         # -----------------------------
#         a_new, logp, _ = self.actor.sample(s)
#         q1_new, q2_new = self.critic(s, a_new)
#         q_new = torch.min(q1_new, q2_new)

#         actor_loss = (self.alpha * logp - q_new).mean()

#         self.actor_opt.zero_grad()
#         actor_loss.backward()
#         self.actor_opt.step()

#         # -----------------------------
#         # Temperature (α) Update
#         # -----------------------------
#         alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

#         self.alpha_opt.zero_grad()
#         alpha_loss.backward()
#         self.alpha_opt.step()

#         self.alpha = self.log_alpha.exp().detach()

#         # -----------------------------
#         # Soft Target Update
#         # -----------------------------
#         for target_param, param in zip(self.critic_target.parameters(),
#                                       self.critic.parameters()):
#             target_param.data.copy_(
#                 self.tau * param.data + (1 - self.tau) * target_param.data
#             )

#         return {
#             "critic_loss": critic_loss.item(),
#             "actor_loss": actor_loss.item(),
#             "alpha_loss": alpha_loss.item(),
#             "alpha": self.alpha.item()
#         }

#     # ============================================================
#     # Save / Load Checkpoint
#     # ============================================================
#     def save_checkpoint(self, path, replay_buffer=None, step=None):
#         checkpoint = {
#             "actor": self.actor.state_dict(),
#             "critic": self.critic.state_dict(),
#             "critic_target": self.critic_target.state_dict(),

#             "actor_opt": self.actor_opt.state_dict(),
#             "critic_opt": self.critic_opt.state_dict(),
#             "alpha_opt": self.alpha_opt.state_dict(),

#             "log_alpha": self.log_alpha.detach().cpu(),

#             "gamma": self.gamma,
#             "tau": self.tau,
#             "target_entropy": self.target_entropy,
#         }

#         if replay_buffer is not None:
#             checkpoint["replay_buffer"] = {
#                 "state": replay_buffer.state[:replay_buffer.size],
#                 "action": replay_buffer.action[:replay_buffer.size],
#                 "reward": replay_buffer.reward[:replay_buffer.size],
#                 "next_state": replay_buffer.next_state[:replay_buffer.size],
#                 "done": replay_buffer.done[:replay_buffer.size],
#                 "ptr": replay_buffer.ptr,
#                 "size": replay_buffer.size,
#             }

#         if step is not None:
#             checkpoint["step"] = step

#         torch.save(checkpoint, path)
#         print(f"[✓] Saved checkpoint → {path}")

#     def load_checkpoint(self, path, replay_buffer=None):
#         checkpoint = torch.load(path, map_location=self.device)

#         self.actor.load_state_dict(checkpoint["actor"])
#         self.critic.load_state_dict(checkpoint["critic"])
#         self.critic_target.load_state_dict(checkpoint["critic_target"])

#         self.actor_opt.load_state_dict(checkpoint["actor_opt"])
#         self.critic_opt.load_state_dict(checkpoint["critic_opt"])
#         self.alpha_opt.load_state_dict(checkpoint["alpha_opt"])

#         self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
#         self.alpha = self.log_alpha.exp().detach()

#         self.gamma = checkpoint["gamma"]
#         self.tau = checkpoint["tau"]
#         self.target_entropy = checkpoint["target_entropy"]

#         if replay_buffer is not None and "replay_buffer" in checkpoint:
#             data = checkpoint["replay_buffer"]
#             n = data["size"]

#             replay_buffer.state[:n] = data["state"]
#             replay_buffer.action[:n] = data["action"]
#             replay_buffer.reward[:n] = data["reward"]
#             replay_buffer.next_state[:n] = data["next_state"]
#             replay_buffer.done[:n] = data["done"]

#             replay_buffer.ptr = data["ptr"]
#             replay_buffer.size = n

#         print(f"[✓] Loaded checkpoint ← {path}")