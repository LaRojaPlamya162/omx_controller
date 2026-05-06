import numpy as np
import torch
import os
import pandas as pd
import numpy as np

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0


        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, s, a, r, s_, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s_
        self.done[self.ptr] = d

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)


        return (
            torch.as_tensor(self.state[idx], device=self.device),
            torch.as_tensor(self.action[idx], device=self.device),
            torch.as_tensor(self.reward[idx], device=self.device),
            torch.as_tensor(self.next_state[idx], device=self.device),
            torch.as_tensor(self.done[idx], device=self.device),
        )

    def __len__(self):
        return self.size
    # ===============================
    # SAVE BUFFER
    # ===============================
    def save(self, path):
        data = {
            "state": self.state[:self.size],
            "action": self.action[:self.size],
            "reward": self.reward[:self.size],
            "next_state": self.next_state[:self.size],
            "done": self.done[:self.size],
            "ptr": self.ptr,
            "size": self.size
        }

        torch.save(data, path)
        #print(f"ReplayBuffer saved to {path}")

    # ===============================
    # LOAD BUFFER
    # ===============================
    def load(self, path):
        if not os.path.exists(path):
            print("ReplayBuffer file not found")
            return

        data = torch.load(path, weights_only=False)

        size = data["size"]

        self.state[:size] = data["state"]
        self.action[:size] = data["action"]
        self.reward[:size] = data["reward"]
        self.next_state[:size] = data["next_state"]
        self.done[:size] = data["done"]

        self.ptr = data["ptr"]
        self.size = size

        print(f"ReplayBuffer loaded from {path} with {self.size} samples")


def fill_replay_buffer_from_dataframe(
    replay_buffer: ReplayBuffer,
    df: pd.DataFrame,
    verbose: bool = True
):
    
    required_cols = [
        # State
        's1','s2','s3','s4','s5','g_s','rb_x','rb_y','rb_z',
        # Action
        'a1','a2','a3','a4','a5','g_a',
        # Next state
        'ns1','ns2','ns3','ns4','ns5','ng_s','nrb_x','nrb_y','nrb_z',
        # Reward & Done
        'reward', 'done'
    ]
    
    # Kiểm tra cột
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu các cột sau trong DataFrame: {missing}")

    states      = df[['s1','s2','s3','s4','s5','g_s','rb_x','rb_y','rb_z']].values.astype(np.float32)
    actions     = df[['a1','a2','a3','a4','a5','g_a']].values.astype(np.float32)
    next_states = df[['ns1','ns2','ns3','ns4','ns5','ng_s','nrb_x','nrb_y','nrb_z']].values.astype(np.float32)
    rewards     = df['reward'].values.astype(np.float32).reshape(-1, 1)
    dones       = df['done'].values.astype(np.float32).reshape(-1, 1)

    num_transitions = len(df)
    
    if verbose:
        print(f"Đang push {num_transitions:,} transitions vào ReplayBuffer...")

    for i in range(num_transitions):
        replay_buffer.push(
            s   = states[i],
            a   = actions[i],
            r   = rewards[i],
            s_  = next_states[i],
            d   = dones[i]
        )

    if verbose:
        print(f"✅ Đã push xong {replay_buffer.size:,}/{replay_buffer.capacity} transitions "
              f"(ptr = {replay_buffer.ptr})")
    return replay_buffer