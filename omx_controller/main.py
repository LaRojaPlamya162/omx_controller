import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from omx_controller.components.utils import concat_dataset, dataset_length
from omx_controller.models.SAC.replay_buffer import ReplayBuffer, fill_replay_buffer_from_dataframe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===== Tools =====
# df = df[condition].index -> filter dataframe if meets condition
# indices = df[condition].index -> index of element in dataframe if meets condition
# stats = df[(condition)].describe() -> stats of dataframe like: mean, std, max, min, ...(can be filtered by condition)

# ===== Test dataset =====
########## SAC ##########
sac_dir = "omx_controller/models/SAC/logs_4"
sac_files = [os.path.join(sac_dir, f"log_{i}.csv") for i in range(1,19)]
sac_df = concat_dataset(files = sac_files, col_names = ["reward", "distance"])
#print(f"Intial size: {len(sac_df)}")
sac_df = sac_df[sac_df['distance'] <= 3.5]
#print(f"Last size: {len(sac_df)}")
########## BC ##########
bc_dir = "omx_controller/models/BC/logs_3"
bc_files = [os.path.join(bc_dir, f"log_{i}.csv") for i in range(1,4)]
bc_df = concat_dataset(files = bc_files, col_names = ["reward", "distance"])
########## IQL ##########
iql_dir = "omx_controller/models/IQL/logs_2"
iql_files = [os.path.join(iql_dir, f"log_{i}.csv") for i in range(2,6)]
iql_df = concat_dataset(files = iql_files, col_names=["reward", "distance"])
iql_df['distance'] = pd.to_numeric(iql_df['distance'], errors='coerce')
iql_df['reward'] = pd.to_numeric(iql_df['reward'], errors='coerce')
########## BC to SAC ##########
bc_to_sac_dir = "omx_controller/models/BC_to_SAC/logs_1"
bc_to_sac_files = [os.path.join(bc_to_sac_dir, f"log_{i}.csv") for i in range(1,8)]
bc_to_sac_df = concat_dataset(files = bc_to_sac_files, col_names=['reward', 'distance'])
bc_to_sac_df = pd.concat([bc_df, bc_to_sac_df], ignore_index=True)
########### Plot ###########
#plt.plot(bc_to_sac_df["reward"])
#plt.scatter(bc_to_sac_df['distance'], bc_to_sac_df['reward'], s= 1)
# plt.xlabel("Distance")
# plt.ylabel("Reward")
# plt.title("BC to SAC")
# plt.grid(True)
# plt.show()  
########## Stats ###########
stats = bc_to_sac_df['reward'].describe()
print(stats)










# ===== Test replay =====
# required_fields = [
#         "s1","s2","s3","s4","s5","g_s","rb_x","rb_y","rb_z",
#         "a1","a2","a3","a4","a5","g_a",
#         "ns1","ns2","ns3","ns4","ns5","ng_s","nrb_x","nrb_y","nrb_z",
#         "reward","done"
#     ]
# bc_files = [os.path.join("omx_controller/models/BC/logs_3", f"log_{i}.csv") for i in range(1,4)]
# df = concat_dataset(files=bc_files, col_names=required_fields)

# replay = ReplayBuffer(state_dim = 9, action_dim = 6, capacity = 1000000, device = DEVICE)
# print(f"Intial size: {len(replay)}")
# replay = fill_replay_buffer_from_dataframe(replay, df)
# print(f"Last size: {len(replay)}")
