# #!/usr/bin/env python3

import os
import sys
import time
import csv
import torch
import numpy as np

# ROS2 Libs
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.executors import MultiThreadedExecutor

# Component Libs (Giữ nguyên cấu trúc import của bạn)
from omx_controller.models.BC.bc_model import BCPolicy
from omx_controller.models.SAC.sac_model import SACAgent
from omx_controller.models.SAC.replay_buffer import ReplayBuffer

class RealRobotController(Node):

    def __init__(self):
        super().__init__('omx_real_controller')

        # 1. Cấu hình khớp (6 khớp bao gồm cả gripper)
        self.joint_names = [
            'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'gripper_joint_1'
        ]
        self.num_joints = len(self.joint_names)

        # 2. Khai báo Publisher & Subscriber
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )
        self.joint_subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )
        self.joint_min = np.array([-2.8, -1.7, -1.5, -1.7, -2.8, 0.0]) 
        self.joint_max = np.array([ 2.8,  1.5,  1.7,  1.7,  2.8, 0.019])
        # 3. Trạng thái robot
        self.current_joint_positions = [0.0] * self.num_joints
        self.initial_robot_pose = None
        self.joint_received = False
        self.resetting = False
        self.reset_state = 'none' # 'none' hoặc 'resetting'
        self.reset_sent = False
        self.reset_start_time = None
        # 4. Tải Model (6 đầu vào, 6 đầu ra)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # BC Model
        self.bc_model = BCPolicy(state_dim=self.num_joints, action_dim=self.num_joints)
        bc_path = "src/omx_controller/omx_controller/models/BC/bc_model.pth"
        if os.path.exists(bc_path):
            self.bc_model.load_state_dict(torch.load(bc_path, map_location=self.device))
            self.bc_model.eval()
            self.get_logger().info("Loaded BC Model")

        # SAC Agent
        self.agent = SACAgent(state_dim=self.num_joints, action_dim=self.num_joints)
        sac_path = "src/omx_controller/omx_controller/models/SAC/SAC.pth"
        if os.path.exists(sac_path):
            self.agent.load_checkpoint(sac_path)
            self.get_logger().info("Loaded SAC Model")

        # 5. Logging & CSV
        self.episode = 0
        self.episode_step = 0
        self.log_path = "src/omx_controller/omx_controller/models/SAC/real_robot_log.csv"
        self.csv_file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'episode', 'step',
            'j1', 'j2', 'j3', 'j4', 'j5', 'g1', # State
            'a1', 'a2', 'a3', 'a4', 'a5', 'ag1'  # Action
        ])

        # 6. Timer điều khiển (20Hz)
        # self.control_timer = self.create_timer(0.05, self.control_step)
        self.control_timer = self.create_timer(0.1, self.control_step)
        self.get_logger().info("Real Robot Controller Started (20Hz)")

    def joint_state_callback(self, msg):
        """Cập nhật trạng thái khớp từ robot thật"""
        # Kiểm tra xem tất cả tên khớp có trong msg không
        if all(name in msg.name for name in self.joint_names):
            for i, name in enumerate(self.joint_names):
                idx = msg.name.index(name)
                self.current_joint_positions[i] = msg.position[idx]
            self.joint_received = True
    def send_trajectory(self, target_positions, duration_sec=0.15):
    # def send_trajectory(self, target_positions, duration_sec=0.1):
        """Gửi lệnh quỹ đạo tới robot"""
        msg = JointTrajectory()
        msg.joint_names = self.joint_names
        msg.header.stamp = self.get_clock().now().to_msg()

        point = JointTrajectoryPoint()
        point.positions = [float(p) for p in target_positions]

        # max_vel = 0.5  # rad/s
        # delta = target - current
        # vel = np.clip(delta / duration_sec, -max_vel, max_vel)
        # point.velocities = vel.tolist()
        point.velocities = [0.0] * self.num_joints

        # Duration nên lớn hơn một chút so với chu kỳ timer (0.05s) để mượt
        point.time_from_start = Duration(seconds=0, nanoseconds=int(duration_sec*1e9)).to_msg()

        msg.points.append(point)
        self.arm_publisher.publish(msg)
    def normalize_state(self, state):
        """Chuyển Radian -> [-1, 1] để đưa vào Model"""
        state = np.array(state)
        norm_state = 2.0 * (state - self.joint_min) / (self.joint_max - self.joint_min + 1e-6) - 1.0
        return np.clip(norm_state, -1.0, 1.0)

    def denormalize_action(self, action):
        """Chuyển [-1, 1] từ Model -> Radian để gửi cho Robot"""
        action = np.array(action)
        real_action = (action + 1.0) * 0.5 * (self.joint_max - self.joint_min) + self.joint_min
        return real_action

    def control_step(self):
        if not self.joint_received:
            return

        # Lưu tư thế ban đầu để reset
        if self.initial_robot_pose is None:
            self.initial_robot_pose = list(self.current_joint_positions)
            self.get_logger().info(f"Captured Home Pose: {self.initial_robot_pose}")
            return
    #     if self.reset_state == 'resetting':

    # # Chỉ gửi 1 lần duy nhất
    #         if not self.reset_sent:
    #             self.get_logger().info("Sending reset trajectory...")
    #             self.send_trajectory(self.initial_robot_pose, duration_sec=2.5)
    #             self.reset_sent = True
    #             return

    #         # Sau khi đã gửi, chỉ kiểm tra vị trí
    #         diff = np.abs(
    #             np.array(self.current_joint_positions) - 
    #             np.array(self.initial_robot_pose)
    #         )

    #         if np.all(diff < 0.02):   # tăng tolerance lên 0.02
    #             self.get_logger().info("Reset Complete. Starting new episode...")
    #             self.reset_state = 'none'
    #             self.reset_sent = False
    #             self.episode += 1
    #             self.episode_step = 0

    #         return
        # Logic Reset (Dùng khi kết thúc episode hoặc gặp lỗi)
        # if self.reset_state == 'resetting':
        #     self.send_trajectory(self.initial_robot_pose, duration_sec=1.5)
            
        #     # Kiểm tra xem đã về gần vị trí home chưa
        #     diff = np.abs(np.array(self.current_joint_positions) - np.array(self.initial_robot_pose))
        #     if np.all(diff < 0.01):
        #         self.get_logger().info("Reset Complete. Starting new episode...")
        #         self.reset_state = 'none'
        #         self.episode += 1
        #         self.episode_step = 0
        #     return
        state_norm = self.normalize_state(self.current_joint_positions)
        state_tensor = torch.tensor(state_norm, dtype=torch.float32).to(self.device).unsqueeze(0)
        # ===== CHẠY INFERENCE (Dự đoán hành động) =====
        #state_tensor = torch.tensor(self.current_joint_positions, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            action_tensor, _ = self.agent.actor.sample(state_tensor) # Dùng sample để giống lúc train
            action = action_tensor.squeeze(0).cpu().numpy()
        # step_size = 0.05 
        step_size = 0.02
        target_positions = np.array(self.current_joint_positions) + action * step_size
        target_positions = np.clip(target_positions, self.joint_min, self.joint_max)
        # Gửi lệnh điều khiển
        self.send_trajectory(target_positions, duration_sec=0.1)

        # Lưu Log
        if self.episode_step % 50 == 0:
            self.get_logger().info(f"Step {self.episode_step}:")
            self.get_logger().info(f"  Current (Rad): {self.current_joint_positions}")
            self.get_logger().info(f"  Action (Model): {action}")
            self.get_logger().info(f"  Target (Rad): {target_positions}")
        
        self.episode_step += 1

        # Ví dụ: Tự động reset sau 200 bước (10 giây)
        if self.episode_step % 200 == 0: 
            self.get_logger().warn("Episode limit reached. Resetting...")
            self.reset_state = 'resetting'

    def shutdown(self):
        """Đóng file và dừng robot"""
        self.get_logger().info("Shutting down...")
        self.csv_file.close()
        # Dừng robot tại chỗ
        self.send_trajectory(self.current_joint_positions, duration_sec=2.5)

def main():
    rclpy.init()
    node = RealRobotController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



#!/usr/bin/env python3

# import os
# import numpy as np
# import torch
# import rclpy

# from rclpy.node import Node
# from rclpy.duration import Duration
# from rclpy.executors import MultiThreadedExecutor

# from sensor_msgs.msg import JointState
# from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# from omx_controller.models.SAC.sac_model import SACAgent


# class OMXRealController(Node):

#     def __init__(self):
#         super().__init__('omx_real_controller')

#         # =========================
#         # Joint Configuration
#         # =========================
#         self.joint_names = [
#             'joint1', 'joint2', 'joint3',
#             'joint4', 'joint5', 'gripper_joint_1'
#         ]
#         self.num_joints = len(self.joint_names)

#         # 🔴 CẬP NHẬT ĐÚNG LIMIT THEO ROBOT THẬT
#         self.joint_min = np.array([-2.8, -1.7, -1.5, -1.7, -2.8, 0.0])
#         self.joint_max = np.array([ 2.8,  1.5,  1.7,  1.7,  2.8, 0.019])

#         # Velocity limit (rad/s)
#         self.max_velocity = 0.5

#         # Increment step scaling
#         self.step_scale = 0.02   # an toàn cho hardware

#         # =========================
#         # ROS2 Pub/Sub
#         # =========================
#         self.traj_pub = self.create_publisher(
#             JointTrajectory,
#             '/arm_controller/joint_trajectory',
#             10
#         )

#         self.create_subscription(
#             JointState,
#             '/joint_states',
#             self.joint_callback,
#             10
#         )

#         # =========================
#         # Robot State
#         # =========================
#         self.current_positions = np.zeros(self.num_joints)
#         self.initial_pose = None
#         self.joint_received = False

#         self.resetting = False

#         # =========================
#         # Load SAC Model
#         # =========================
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.agent = SACAgent(
#             state_dim=self.num_joints,
#             action_dim=self.num_joints
#         )

#         sac_path = "src/omx_controller/omx_controller/models/SAC/SAC.pth"
#         if os.path.exists(sac_path):
#             self.agent.load_checkpoint(sac_path)
#             self.get_logger().info("Loaded SAC model")

#         # =========================
#         # Control Timer (10Hz)
#         # =========================
#         self.control_period = 0.5
#         self.timer = self.create_timer(
#             self.control_period,
#             self.control_loop
#         )

#         self.get_logger().info("OMX Real Controller started (10Hz, safe mode)")

#     # ==========================================================
#     # Joint Callback
#     # ==========================================================
#     def joint_callback(self, msg):

#         if not all(name in msg.name for name in self.joint_names):
#             return

#         for i, name in enumerate(self.joint_names):
#             idx = msg.name.index(name)
#             self.current_positions[i] = msg.position[idx]

#         self.joint_received = True

#     # ==========================================================
#     # Normalize state [-1,1]
#     # ==========================================================
#     def normalize(self, state):
#         norm = 2.0 * (state - self.joint_min) / \
#                (self.joint_max - self.joint_min + 1e-6) - 1.0
#         return np.clip(norm, -1.0, 1.0)

#     # ==========================================================
#     # Control Loop
#     # ==========================================================
#     def control_loop(self):

#         if not self.joint_received:
#             return

#         # Capture initial pose
#         if self.initial_pose is None:
#             self.initial_pose = self.current_positions.copy()
#             self.get_logger().info(f"Captured home pose: {self.initial_pose}")
#             return

#         if self.resetting:
#             self.move_to_pose(self.initial_pose, duration=0.5)
#             if np.all(np.abs(self.current_positions - self.initial_pose) < 0.01):
#                 self.get_logger().info("Reset complete")
#                 self.resetting = False
#             return

#         # ==========================
#         # Prepare state
#         # ==========================
#         state_norm = self.normalize(self.current_positions)
#         state_tensor = torch.tensor(
#             state_norm,
#             dtype=torch.float32,
#             device=self.device
#         ).unsqueeze(0)

#         # ==========================
#         # Deterministic inference
#         # ==========================
#         with torch.no_grad():
#             action_tensor, _ = self.agent.actor(state_tensor)

#         action = action_tensor.squeeze(0).cpu().numpy()
#         self.get_logger().info(f"Action: {action}")

#         # NaN safety
#         if np.any(np.isnan(action)):
#             self.get_logger().error("NaN detected in action. Skipping step.")
#             return

#         # ==========================
#         # Incremental control
#         # ==========================
#         delta = action * self.step_scale
#         target = self.current_positions + delta

#         # Clamp joint limits
#         target = np.clip(target, self.joint_min, self.joint_max)

#         # ==========================
#         # Send safe trajectory
#         # ==========================
#         self.move_to_pose(target, duration=0.15)

#     # ==========================================================
#     # Safe trajectory sender
#     # ==========================================================
#     def move_to_pose(self, target, duration=0.15):

#         traj = JointTrajectory()
#         traj.header.stamp = self.get_clock().now().to_msg()
#         traj.joint_names = self.joint_names

#         point = JointTrajectoryPoint()
#         point.positions = target.tolist()

#         # Compute velocity safely
#         delta = target - self.current_positions
#         vel = delta / duration
#         vel = np.clip(vel, -self.max_velocity, self.max_velocity)

#         point.velocities = vel.tolist()
#         point.time_from_start = Duration(
#             seconds=0,
#             nanoseconds=int(duration * 1e9)
#         ).to_msg()

#         traj.points.append(point)
#         self.traj_pub.publish(traj)

#     # ==========================================================
#     # Shutdown safety
#     # ==========================================================
#     def shutdown(self):
#         self.get_logger().info("Stopping robot safely...")
#         self.move_to_pose(self.current_positions, duration=0.5)


# # ==============================================================
# # Main
# # ==============================================================

# def main():
#     rclpy.init()
#     node = OMXRealController()

#     executor = MultiThreadedExecutor()
#     executor.add_node(node)

#     try:
#         executor.spin()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.shutdown()
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()