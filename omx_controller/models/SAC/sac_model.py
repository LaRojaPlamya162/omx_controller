from omx_controller.models.SAC.sac_network import Actor, Critic
from omx_controller.models.BC.bc_model import BCPolicy
from omx_controller.models.IQL.iql_model import IQLAgent
from omx_controller.components.utils import get_action_max_min
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BC_DIR = "src/omx_controller/omx_controller/models/BC"
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
action_max, action_min = get_action_max_min()
class SACAgent:
    def __init__(self, state_dim, action_dim):
        #self.actor = Actor(state_dim, action_dim)
        self.actor = Actor(state_dim, action_dim, action_min=action_min, action_max=action_max)
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
            #print(f"[✓] Saved to {path}")

    
    def load_checkpoint(self, path):
        #ckpt_bc = torch.load(os.path.join(BC_DIR, "bc_model_v2_squashed_real.pth"), map_location=DEVICE)
        ckpt = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(ckpt["actor"])
        self.q1.load_state_dict(ckpt["q1"])
        self.q2.load_state_dict(ckpt["q2"])
        self.q1_target.load_state_dict(ckpt["q1_target"])
        self.q2_target.load_state_dict(ckpt["q2_target"])

        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.q1_opt.load_state_dict(ckpt["q1_opt"])
        self.q2_opt.load_state_dict(ckpt["q2_opt"])

        self.log_alpha.data.copy_(ckpt['log_alpha'].to(DEVICE))
        self.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        print(f"[✓] Loaded from {path}")

def initialize_sac_from_bc(sac_agent: SACAgent, bc_checkpoint_path: str, state_dim: int = 9, action_dim: int = 6):
    """
    Load BC policy vào SAC Actor (warm-start)
    """
    print(f"[BC → SAC] Loading BC weights from: {bc_checkpoint_path}")

    # 1. Load BC model
    bc_model = BCPolicy(state_dim, action_dim, hidden_dim=256).to(DEVICE)
    checkpoint = torch.load(bc_checkpoint_path, map_location=DEVICE, weights_only=True)
    bc_model.load_state_dict(checkpoint["model_state_dict"])
    bc_model.eval()

    # 2. Copy weights sang SAC Actor (backbone + mean + log_std)
    sac_actor = sac_agent.actor

    # Copy backbone
    sac_actor.backbone.load_state_dict(bc_model.backbone.state_dict())
    
    # Copy mean head
    sac_actor.mean.load_state_dict(bc_model.mean.state_dict())
    
    # Copy log_std head (SAC có scaling tanh khác một chút, nhưng rất ổn để warm-start)
    sac_actor.log_std.load_state_dict(bc_model.log_std.state_dict())

    print("[✓] SAC Actor đã được initialize từ BC policy!")
    print("    → Backbone, mean, log_std đã copy thành công")
    print("    → SAC sẽ tiếp tục fine-tune từ policy khá tốt của BC")

    return sac_agent
def initialize_sac_from_iql(sac: SACAgent, iql_checkpoint_path: str, state_dim:int = 9, action_dim: int = 6):
    """
    Load weights từ IQL checkpoint (đường dẫn) vào SAC để fine-tune.
    - Tự động infer state_dim / action_dim từ SAC actor
    - Tạo IQL tạm thời chỉ để load checkpoint rồi copy weights
    - Không làm thay đổi bất kỳ thứ gì khác của SAC
    """
    DEVICE = next(sac.actor.parameters()).device
    
    # === Infer dimensions từ SAC (không cần hardcode 9/6) ===
    # state_dim = sac.actor.backbone[0].in_features   # Linear(state_dim → hidden)
    # action_dim = sac.actor.mean.out_features        # Linear(hidden → action_dim)
    
    # === Tạo IQL tạm thời và load checkpoint ===
    iql = IQLAgent(state_dim=state_dim, action_dim=action_dim, device=DEVICE)
    iql.load_checkpoint(iql_checkpoint_path)
    
    # === 1. Copy Q networks (giống hệt) ===
    sac.q1.load_state_dict(iql.q1.state_dict())
    sac.q2.load_state_dict(iql.q2.state_dict())
    sac.q1_target.load_state_dict(iql.q1_target.state_dict())
    sac.q2_target.load_state_dict(iql.q2_target.state_dict())
    
    # === 2. Copy Actor ← Policy (map keys vì tên layer khác nhau) ===
    pi_dict = iql.pi.state_dict()
    actor_dict = sac.actor.state_dict()
    
    # Backbone / net
    actor_dict['backbone.0.weight'] = pi_dict['net.0.weight']
    actor_dict['backbone.0.bias']   = pi_dict['net.0.bias']
    actor_dict['backbone.2.weight'] = pi_dict['net.2.weight']
    actor_dict['backbone.2.bias']   = pi_dict['net.2.bias']
    
    # Mean & log_std
    actor_dict['mean.weight']      = pi_dict['mean.weight']
    actor_dict['mean.bias']        = pi_dict['mean.bias']
    actor_dict['log_std.weight']   = pi_dict['log_std.weight']
    actor_dict['log_std.bias']     = pi_dict['log_std.bias']
    
    sac.actor.load_state_dict(actor_dict)
    
    # === 3. Reset alpha về giá trị hợp lý (SAC v2) ===
    # Khởi đầu alpha ≈ 0.2 thường ổn định hơn khi fine-tune từ IQL
    # with torch.no_grad():
    #     sac.log_alpha.data.copy_(torch.log(torch.tensor([0.2], device=DEVICE)))
    
    return sac