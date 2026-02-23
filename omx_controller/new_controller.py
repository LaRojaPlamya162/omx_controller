#!/usr/bin/env python3

# ===== System Lib =====
import sys
import time
import torch
import csv
import os
import numpy as np
# ===== ROS2 Lib =====
from control_msgs.action import GripperCommand
from geometry_msgs.msg import Pose
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ros_gz_interfaces.srv import DeleteEntity, SpawnEntity
from ros_gz_interfaces.msg import Entity
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from tf2_ros import Buffer, TransformListener
# ===== Component Lib =====
from omx_controller.models.BC.bc_model import BCPolicy
from omx_controller.components.reward import RewardFunction
class Controller(Node):

    def __init__(self):
        super().__init__('keyboard_controller')

        # QoS profile for reliable subscriptions
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Publisher for arm joint control
        self.arm_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10
        )

        # Action client for GripperCommand
        self.gripper_client = ActionClient(
            self, GripperCommand, '/gripper_controller/gripper_cmd'
        )

        # Subscriber for joint states
        self.subscription = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10
        )

        # Subscriber to get ball pose
        self.ball_sub = self.create_subscription(
            Pose,
            '/cricket_ball/pose',
            self.ball_callback,
            qos
        )
        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Clients for spawn and delete services
        self.spawn_client = self.create_client(SpawnEntity, '/world/empty/create')
        self.delete_client = self.create_client(DeleteEntity, '/world/empty/remove')

        # Wait for services with retry
        max_attempts = 30
        attempt = 0
        while attempt < max_attempts and rclpy.ok():
            spawn_ready = self.spawn_client.wait_for_service(timeout_sec=10.0)
            delete_ready = self.delete_client.wait_for_service(timeout_sec=10.0)
            if spawn_ready and delete_ready:
                self.get_logger().info('Services /world/empty/create and /world/empty/remove are available.')
                break
            attempt += 1
            self.get_logger().warn(f'Services not available yet (attempt {attempt}/{max_attempts}) - spawn: {spawn_ready}, delete: {delete_ready}. Retrying in 5 seconds...')
            time.sleep(5.0)
        if attempt == max_attempts:
            self.get_logger().error('Services not available after max attempts. Node may not function properly.')

        # Wait for gripper action server
        while not self.gripper_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().warn('Waiting for gripper action server...')
            time.sleep(1.0)

        # Initial states
        self.initial_arm_positions = [0.0] * 5
        self.initial_gripper_position = 0.0
        self.arm_joint_positions = [0.0] * 5
        self.arm_joint_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
        ]
        self.gripper_position = 0.0
        self.gripper_max = 1.1
        self.gripper_min = 0.0
        #self.initial_ball_pose = [0.0, 2.0, 1.0]  # Consistent with spawn position
        self.joint_received = False
        self.initial_omx_pose = None
        self.ball_pos = [0.2, 0.2, 0.0]  # Default ball position
        self.joint_pos = None
        # Control parameters
        self.max_delta = 0.02
        self.gripper_delta = 0.1
        self.last_command_time = time.time()
        self.command_interval = 0.02

        # Gripper send throttling
        self.last_gripper_send_time = 0.0
        self.gripper_send_interval = 0.2

        # Logging throttle (every 1 second)
        self.last_joint_log_time = 0.0
        self.last_arm_log_time = 0.0
        self.last_gripper_log_time = 0.0
        self.last_ball_log_time = 0.0
        self.log_interval = 1.0

        # Model setup
        self.model = BCPolicy(state_dim=6, action_dim=6)
        self.model.load_state_dict(torch.load("src/omx_controller/omx_controller/models/BC/bc_model.pth", weights_only=True))
        self.model.eval()

        # Control / Logging variables
        self.prev_arm_positions = None
        self.prev_gripper_position = None
        self.last_action_arm = None
        self.last_action_gripper = None
        self.resetting = False
        self.new_episode_ready = True
        self.episode = 0
        self.timestep = 0
        self.episode_step = 0
        self.reset_state = 'none'
        self.ball_reset_in_progress = False
        self.tolerance = 0.001  # Tolerance for pose comparison

        # CSV logging
        self.csv_file = open("src/omx_controller/omx_controller/models/BC/bc_log.csv", "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'episode', 'timestep',
            's1', 's2', 's3', 's4', 's5', 'g_s',
            'n_s1', 'n_s2', 'n_s3', 'n_s4', 'n_s5', 'n_g_s',
            'a1', 'a2', 'a3', 'a4', 'a5', 'g_a',
            'jp_x', 'jp_y', 'jp_z',
            'bp_x', 'bp_y', 'bp_z',
            'reward', 'done'
        ])

        # Create timer for control loop (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_step)

    def is_pose_near_initial(self):
        if self.initial_omx_pose is None:
            return False
        initial_arm = self.initial_omx_pose[:5]
        initial_gripper = self.initial_omx_pose[5]
        arm_close = all(abs(a - b) < self.tolerance for a, b in zip(self.arm_joint_positions, initial_arm))
        gripper_close = abs(self.gripper_position - initial_gripper) < self.tolerance
        return arm_close and gripper_close

    def joint_state_callback(self, msg):
        if set(self.arm_joint_names).issubset(set(msg.name)):
            for i, joint in enumerate(self.arm_joint_names):
                index = msg.name.index(joint)
                self.arm_joint_positions[i] = msg.position[index]

        if 'rh_r1_joint' in msg.name:
            index = msg.name.index('rh_r1_joint')
            self.gripper_position = msg.position[index]

        self.joint_received = True

        # Throttled logging
        current_time = time.time()
        if current_time - self.last_joint_log_time >= self.log_interval:
            self.get_logger().info(
                f'Received joint states: {self.arm_joint_positions}, '
                f'Gripper: {self.gripper_position}'
            )
            self.last_joint_log_time = current_time

    def ball_callback(self, msg):
        pos = msg.position
        x, y, z = pos.x, pos.y, pos.z
        self.ball_pos = [x, y, z]

        # Throttled logging
        current_time = time.time()
        if current_time - self.last_ball_log_time >= self.log_interval:
            self.get_logger().info(f"Ball position: {x:.3f}, {y:.3f}, {z:.3f}")
            self.last_ball_log_time = current_time

    def send_arm_command(self, arm_pos):
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = arm_pos
        arm_point.time_from_start = Duration(seconds=0, nanoseconds=50000000).to_msg()
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)

        # Throttled logging
        current_time = time.time()
        if current_time - self.last_arm_log_time >= self.log_interval:
            self.get_logger().info(f'Arm command sent: {arm_pos}')
            self.last_arm_log_time = current_time

    def send_gripper_command(self, gripper_pos):
        current_time = time.time()
        if current_time - self.last_gripper_send_time < self.gripper_send_interval:
            return

        self.last_gripper_send_time = current_time

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = gripper_pos
        goal_msg.command.max_effort = 10.0

        if not self.gripper_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn('Gripper action server not available')
            return

        self.gripper_client.send_goal_async(goal_msg)

    def control_step(self):
        if not self.joint_received:
            return

        if self.initial_omx_pose is None:
            self.initial_omx_pose = self.arm_joint_positions + [self.gripper_position]
            self.get_logger().info("Initial OMX pose captured!")

        # ===== RESET STATE MACHINE =====
        if self.reset_state == 'reset_robot':
            self.reset_omx_pose()
            if self.is_pose_near_initial():
                self.get_logger().info("Robot reset done")
                self.reset_state = 'reset_ball'
            return

        elif self.reset_state == 'reset_ball':
            if not self.ball_reset_in_progress:
                self.ball_reset_in_progress = True
                self.reset_ball()
            return

        if self.resetting or not self.new_episode_ready:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
            'world',                 # frame gốc
            'end_effector_link',     # khớp cuối
            rclpy.time.Time()
            )
            
            ee_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
        except Exception:
            ee_pos = [np.nan, np.nan, np.nan]
        
        self.joint_pos = ee_pos
        # ===== MODEL CONTROL =====
        current_state_list = self.arm_joint_positions + [self.gripper_position]
        # ===== Reward =====
        reward_fn = RewardFunction(
            self.ball_pos,
            self.joint_pos,
            self.timestep
        )
        reward = reward_fn.reward
        done = reward_fn.done
        if self.last_action_arm is not None:
            action_log = self.last_action_arm + [self.last_action_gripper]
            row = (
                [self.episode] +
                [self.timestep] +
                self.prev_arm_positions +
                [self.prev_gripper_position] +
                current_state_list +
                action_log +
                self.joint_pos + 
                self.ball_pos +
                [reward] +
                [done]
            )
            self.writer.writerow(row)
            if self.timestep % 50 == 0:
                self.csv_file.flush()

            self.timestep += 1
            self.episode_step += 1
            if self.timestep % 1000 == 0:
                self.get_logger().info("Episode done -> start reset")
                self.reset_state = 'reset_robot'
                self.resetting = True
                self.new_episode_ready = False
                self.episode += 1
                self.timestep = 0
                return

        state_tensor = torch.tensor(current_state_list, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            action_tensor = self.model.act(state_tensor, deterministic=True)

        action = action_tensor.squeeze(0).cpu().numpy()
        self.last_action_arm = action[:5].tolist()
        self.last_action_gripper = float(action[5])

        self.send_arm_command(self.last_action_arm)
        self.send_gripper_command(self.last_action_gripper)

        self.prev_arm_positions = self.arm_joint_positions.copy()
        self.prev_gripper_position = self.gripper_position

    def reset_omx_pose(self):
        if self.initial_omx_pose is None:
            return

        arm_pos = self.initial_omx_pose[:5]
        gripper_pos = float(self.initial_omx_pose[5])

        self.send_arm_command(arm_pos)
        self.send_gripper_command(gripper_pos)

    def reset_ball(self):
        self.get_logger().info("Deleting ball...")

        delete_req = DeleteEntity.Request()
        delete_req.entity = Entity()
        delete_req.entity.name = 'cricket_ball'
        delete_req.entity.type = 2  # EntityType.MODEL

        future = self.delete_client.call_async(delete_req)
        future.add_done_callback(self.delete_done_callback)

    def delete_done_callback(self, future):
        try:
            result = future.result()
            if result is None:
                self.get_logger().error("Delete returned None")
                return

            if result.success:
                self.get_logger().info("Delete success")
            else:
                self.get_logger().warn("Delete failed, spawning anyway")

        except Exception as e:
            self.get_logger().error(f"Delete exception: {e}")

        self.spawn_ball()

    def spawn_ball(self):
        self.get_logger().info("Spawning ball...")

        spawn_req = SpawnEntity.Request()
        spawn_req.entity_factory.name = 'cricket_ball'
        spawn_req.entity_factory.allow_renaming = False

        model_path = os.path.expanduser(
            '~/.gz/fuel/fuel.gazebosim.org/openrobotics/models/cricket%20ball/3/model.sdf'
        )

        with open(model_path) as f:
            spawn_req.entity_factory.sdf = f.read()

        spawn_req.entity_factory.pose.position.x = self.ball_pos[0]
        spawn_req.entity_factory.pose.position.y = self.ball_pos[1]
        spawn_req.entity_factory.pose.position.z = self.ball_pos[2]

        spawn_req.entity_factory.relative_to = "world"

        future = self.spawn_client.call_async(spawn_req)
        future.add_done_callback(self.spawn_done_callback)

    def spawn_done_callback(self, future):
        try:
            result = future.result()
            if result is None or not result.success:
                self.get_logger().error("Spawn failed")
                return

            self.get_logger().info("Ball reset complete")

        except Exception as e:
            self.get_logger().error(f"Spawn exception: {e}")
            return

        # Resume training
        self.last_action_arm = None
        self.last_action_gripper = None
        self.prev_arm_positions = self.initial_omx_pose[:5]
        self.prev_gripper_position = self.initial_omx_pose[5]
        #self.prev_arm_positions = self.arm_joint_positions.copy()
        #self.prev_gripper_position = self.gripper_position

        self.reset_state = 'none'
        self.resetting = False
        self.new_episode_ready = True
        self.ball_reset_in_progress = False

def main():
    rclpy.init()
    node = Controller()

    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Shutting down...')
    finally:
        # Close CSV file
        if hasattr(node, 'csv_file'):
            node.csv_file.close()
            print("CSV log file closed.")

        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()