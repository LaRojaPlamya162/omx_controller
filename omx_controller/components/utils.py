import os
import torch
from pathlib import Path
from datasets import load_dataset
import pandas as pd
from collections import defaultdict
import re
LOG_STD_MAX = 2.0
LOG_STD_MIN = -5.0
def get_action_max_min(BASE_PATH = "/root/ros2_ws/src/physical_ai_tools/docker/huggingface/lerobot/LearnDLFromScratch/omx_f_ReallocateObject/data/chunk-000"):
    data_files = [os.path.join(BASE_PATH, f"episode_{i:06d}.parquet") for i in range(20)]
    dataset = load_dataset("parquet", data_files=data_files, split="train")
    dataset = dataset.with_format("torch")
    actions = torch.stack([dataset[i]["action"] for i in range(len(dataset))])
    action_min = actions.min(dim=0)[0]
    action_max = actions.max(dim=0)[0]
    # action_max = action_max.cpu().numpy()
    # action_min = action_min.cpu().numpy()
    return action_max, action_min

def create_log_file(path):
        log_dir = Path(path)
        log_dir.mkdir(parents=True, exist_ok=True)

        existing_logs = list(log_dir.glob("log_*.csv"))

        if not existing_logs:
            next_index = 1
        else:
            indices = []
            for f in existing_logs:
                try:
                    idx = int(f.stem.split("_")[1])
                    indices.append(idx)
                except:
                    pass

            next_index = max(indices) + 1

        log_path = log_dir / f"log_{next_index}.csv"
        print(f"Log path: {log_path}")
        return log_path, next_index

def dataset_length(path):
     dir = Path(path)
     dir.mkdir(parents=True, exist_ok=True)

     existing_files = list(dir.glob("log_*.csv"))
     #print(f"Files: {existing_files}")
     total_len = 0
     if not existing_files:
        return total_len
     else:
        for i in range(1, len(existing_files)+1):
            #print(f"File: {os.path.join(path, f"log_{i}.csv")}")
            df = pd.read_csv(os.path.join(path, f"log_{i}.csv"))
            total_len += len(df['timestep'])
        return total_len

def concat_dataset(files, col_names):
    dfs = []

    for file in files:
        df = pd.read_csv(file)
        dfs.append(df[col_names])

    return pd.concat(dfs, ignore_index=True)

def get_latest_file(path):
    pattern = re.compile(r"^(.*)_(\d+)(\.[^.]+)?$")

    max_idx = -1
    latest_file = None

    for fname in os.listdir(path):
        match = pattern.match(fname)
        if match:
            idx = int(match.group(2))
            if idx > max_idx:
                max_idx = idx
                latest_file = fname

    if latest_file is None:
        return None

    return os.path.join(path, latest_file)

if __name__ == '__main__':
    print(get_latest_file("omx_controller/models/BC_to_SAC/logs_1")) 