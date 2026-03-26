# import os

# files = [
#     "log_1.csv","log_2.csv","log_3.csv","log_4.csv","log_5.csv",
#     "log_6.csv","log_7.csv","log_8.csv","log_9.csv","log_10.csv",
# ]

# source_log_dir = "omx_controller/models/SAC/logs"
# dest_log_dir = "omx_controller/models/SAC/final_logs"

# # index cần xóa (sửa theo bạn)
# idx_to_remove = [8,9,10,23,24,25]

# columns = [
#     'episode','timestep','s1','s2','s3','s4','s5',
#     'g_s','n_s1','n_s2','n_s3','n_s4','n_s5','n_g_s',
#     'a1','a2','a3','a4','a5','g_a',
#     'jp_x','jp_y','jp_z','bp_x','bp_y','bp_z',
#     'reward','done'
# ]

# for file in files:
#     in_path = os.path.join(source_log_dir, file)
#     out_path = os.path.join(dest_log_dir, file)

#     with open(in_path, 'r') as f:
#         lines = f.readlines()

#     fixed_lines = []

#     # giữ header (dòng đầu)
#     fixed_lines.append(','.join(columns) + '\n')
#     idx_to_remove = [8, 9, 10, 23, 24, 25]

#     for line in lines[1:]:
#         parts = line.strip().split(',')

#         # Xóa từ phải sang trái để không bị lệch index
#         for idx in sorted(idx_to_remove, reverse=True):
#             if idx < len(parts):
#                 parts.pop(idx)

#         fixed_lines.append(','.join(parts[:28]) + '\n')
    # for line in lines[1:]:
    #     parts = line.strip().split(',')

    #     # bỏ các value dư
    #     parts = [v for i, v in enumerate(parts) if i not in idx_to_remove]

    #     # cắt về đúng 28 giá trị
    #     parts = parts[:28]

    #     fixed_lines.append(','.join(parts) + '\n')

    # with open(out_path, 'w') as f:
    #     f.writelines(fixed_lines)

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

files = [
    "log_2.csv",
    #"log_5.csv",
    "log_7.csv",
    #"log_9.csv",
    "log_14.csv",
    "log_15.csv"
]

log_dir = "omx_controller/models/SAC/logs_3"

for file in files:
    path = os.path.join(log_dir, file)

    df = pd.read_csv(path)
    distance = df['distance']
    reward = df['reward']

    # plot
    #plt.plot(distance, label = file)
    plt.plot(reward, label=file)

plt.xlabel("Timestep")
#plt.ylabel("Distance")
#plt.title("Distance")
plt.ylabel("Reward")
plt.title("Reward")
plt.legend()
plt.grid()

plt.show()