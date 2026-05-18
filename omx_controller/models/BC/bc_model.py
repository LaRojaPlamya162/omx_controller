import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.distributions as D
from omx_controller.components.utils import get_action_max_min

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
NUM_EPOCHS = 100         
LEARNING_RATE = 1e-3
CLIP_GRAD_NORM = 1.0

BASE_PATH = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/LearnDLFromScratch/omx_f_ReallocateObject/data/chunk-000"
class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        h = self.backbone(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def get_log_prob(self, state, action_norm):

        mean, std = self.forward(state)
        
        raw_action = torch.atanh(torch.clamp(action_norm, -0.9999, 0.9999))
        
        normal = D.Normal(mean, std)
        log_prob = normal.log_prob(raw_action)
        
        # Jacobian correction 
        squash_correction = torch.log(1 - action_norm.pow(2) + 1e-6)
        
        log_prob = log_prob - squash_correction
        return log_prob.sum(dim=-1)

    def act(self, state, deterministic=True):
        with torch.no_grad():
            mean, std = self.forward(state)
            
            if deterministic:
                raw_action = mean
            else:
                normal = D.Normal(mean, std)
                raw_action = normal.sample()
            
            action_norm = torch.tanh(raw_action)   
            return action_norm

# ========================== UTILS ==========================
def compute_stats(dataset):
    states = []
    actions = []

    for i in range(len(dataset)):
        s = dataset[i]["observation.state"]

        pad = torch.zeros(3)
        s = torch.cat([s, pad], dim=0)

        states.append(s)
        actions.append(dataset[i]["action"])

    states = torch.stack(states)
    actions = torch.stack(actions)

    state_mean = states.mean(0)
    state_std = states.std(0) + 1e-6

    action_mean = actions.mean(0)
    action_std = actions.std(0) + 1e-6

    return state_mean, state_std, action_mean, action_std

def load_model(path, state_dim, action_dim):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = BCPolicy(state_dim, action_dim).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ========================== MAIN ==========================
if __name__ == '__main__':
    data_files = [os.path.join(BASE_PATH, f"episode_{i:06d}.parquet") for i in range(20)]
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = dataset.with_format("torch")
    print(dataset)
    state_dim = len(dataset[0]["observation.state"]) + 3
    action_dim = len(dataset[0]["action"])
    actions = torch.stack([dataset[i]["action"] for i in range(len(dataset))])
    action_min = actions.min(dim=0)[0]
    action_max = actions.max(dim=0)[0]

    
    state_mean, state_std, _, _ = compute_stats(dataset)
    state_mean = state_mean.to(DEVICE)
    state_std = state_std.to(DEVICE)
    action_min = action_min.to(DEVICE)
    action_max = action_max.to(DEVICE)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    # Model + Optimizer
    model = BCPolicy(state_dim, action_dim, hidden_dim=256).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    model.train()
    print("Bắt đầu training Behavior Cloning (Squashed Gaussian)...")
    
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = 0
        for batch in dataloader:
            state = batch["observation.state"].to(DEVICE)
            action_real = batch["action"].to(DEVICE)

            # Pad state
            pad = torch.zeros(state.shape[0], 3, device=DEVICE)
            state = torch.cat([state, pad], dim=1)
            
            # Normalize
            state_norm = (state - state_mean) / state_std
            
            action_norm = 2.0 * (action_real - action_min) / (action_max - action_min + 1e-8) - 1.0

            # Calculate loss
            log_prob = model.get_log_prob(state_norm, action_norm)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | Loss: {avg_loss:.6f}")
    
    print("Training hoàn tất!")
    
    os.makedirs("omx_controller/models/BC", exist_ok=True)
    
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "hidden_dim": 256,
    }, "omx_controller/models/BC/bc_model_v2_squashed_real.pth")
    print("Model đã lưu tại: omx_controller/models/BC/bc_model_v2_squashed_real.pth")
    
