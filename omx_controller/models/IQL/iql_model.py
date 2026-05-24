import copy
import os
from omx_controller.models.IQL.iql_network import QNetwork, ValueNet, Policy
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os
from torch.utils.data import DataLoader
from datasets import load_dataset
from omx_controller.components.utils import concat_dataset
from datasets import Dataset
# =========================
# IQL Agent
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class IQLAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device

        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)

        self.v = ValueNet(state_dim).to(device)
        self.pi = Policy(state_dim, action_dim).to(device)

        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=3e-4)
        self.pi_optimizer = torch.optim.Adam(self.pi.parameters(), lr=3e-4)

        self.gamma = 0.99
        self.expectile = 0.7
        self.beta = 3.0
        self.tau = 0.005          # soft update

    def expectile_loss(self, diff):
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return weight * diff.pow(2)

    def update(self, batch):
        s, a, r, s_next, done = [x.to(self.device) for x in batch]
        r = r.unsqueeze(-1)
        done = done.float().unsqueeze(-1)

        # 1. Update V
        with torch.no_grad():
            q1_t = self.q1_target(s, a)
            q2_t = self.q2_target(s, a)
            q_t = torch.min(q1_t, q2_t)

        v = self.v(s)
        v_loss = self.expectile_loss(q_t - v).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # 2. Update Q
        with torch.no_grad():
            target_v = self.v(s_next)
            target_q = r + self.gamma * (1 - done) * target_v

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            adv = torch.min(self.q1(s, a), self.q2(s, a)) - self.v(s)
            weights = torch.exp(adv * self.beta).clamp(max=100.0)

        log_prob = self.pi.log_prob(s, a)        
        pi_loss = -(weights * log_prob).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()

        # 4. Soft update target Q
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "v_loss": v_loss.item(),
            "q_loss": q_loss.item(),
            "pi_loss": pi_loss.item()
        }

    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(), "q2_target": self.q2_target.state_dict(),
            "v": self.v.state_dict(), "pi": self.pi.state_dict(),
        }
        torch.save(checkpoint, path)
        print(f"[✓] Saved at: {path}")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)

        # Load networks
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.q1_target.load_state_dict(checkpoint["q1_target"])
        self.q2_target.load_state_dict(checkpoint["q2_target"])
        self.v.load_state_dict(checkpoint["v"])
        self.pi.load_state_dict(checkpoint["pi"])

        # Set eval mode 
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        self.v.eval()
        self.pi.eval()

        print(f"Loaded IQL checkpoint from {path}")

# =========================
# Training Function
# =========================
def train_iql(iql, dataset, epochs=100, batch_size=512, log_interval=5, 
              save_dir="omx_controller/models/IQL/checkpoints"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    dataset = dataset.with_format("torch")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )

    print(f"Bắt đầu train IQL | Dataset size: {len(dataset)} | Epochs: {epochs}\n")

    for epoch in range(epochs):
        epoch_losses = {"v_loss": 0.0, "q_loss": 0.0, "pi_loss": 0.0}

        for batch_dict in dataloader:
            # State 9D (đã scale)
            s = torch.stack([
                batch_dict["s1"], batch_dict["s2"], batch_dict["s3"],
                batch_dict["s4"], batch_dict["s5"], batch_dict["g_s"],
                batch_dict["rb_x"], batch_dict["rb_y"], batch_dict["rb_z"]
            ], dim=1)

            a = torch.stack([
                batch_dict["a1"], batch_dict["a2"], batch_dict["a3"],
                batch_dict["a4"], batch_dict["a5"], batch_dict["g_a"]
            ], dim=1)

            s_next = torch.stack([
                batch_dict["ns1"], batch_dict["ns2"], batch_dict["ns3"],
                batch_dict["ns4"], batch_dict["ns5"], batch_dict["ng_s"],
                batch_dict["nrb_x"], batch_dict["nrb_y"], batch_dict["nrb_z"]
            ], dim=1)

            r = batch_dict["reward"]
            done = batch_dict["done"]

            losses = iql.update((s, a, r, s_next, done))

            for k in epoch_losses:
                epoch_losses[k] += losses[k]

        # ====================== Logging ======================
        if (epoch + 1) % log_interval == 0:
            avg_v = epoch_losses["v_loss"] / len(dataloader)
            avg_q = epoch_losses["q_loss"] / len(dataloader)
            avg_pi = epoch_losses["pi_loss"] / len(dataloader)
            
            print(f"Epoch {epoch+1:4d} | "
                  f"V_loss: {avg_v:.6f} | "
                  f"Q_loss: {avg_q:.6f} | "
                  f"Pi_loss: {avg_pi:.6f}")

        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            iql.save_checkpoint(f"{save_dir}/iql_epoch_{epoch+1:03d}.pth")

    iql.save_checkpoint(f"{save_dir}/iql_final.pth")
    print("\n=== Training IQL hoàn tất ===")


if __name__ == "__main__":
    # Load dataset
    data_files = [
        f"omx_controller/models/SAC/logs_4/log_{i}.csv" for i in range(1, 19)
    ]
    required_fields = [
        "s1","s2","s3","s4","s5","g_s","rb_x","rb_y","rb_z",
        "a1","a2","a3","a4","a5","g_a",
        "ns1","ns2","ns3","ns4","ns5","ng_s","nrb_x","nrb_y","nrb_z",
        "reward","done"
    ]
    df = concat_dataset(files = data_files, col_names= required_fields)
    dataset = Dataset.from_pandas(df)
    state_dim = 9
    action_dim = 6

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Dataset size: {len(dataset)}\n")

    iql = IQLAgent(state_dim, action_dim, DEVICE)

    train_iql(
        iql=iql,
        dataset=dataset,
        epochs=100,           
        batch_size=512,
        log_interval=5,
        save_dir="omx_controller/models/IQL/checkpoints_2"
    )