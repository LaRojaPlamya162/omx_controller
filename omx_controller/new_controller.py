#!/usr/bin/env python3

# ===== System Lib =====
import select
import sys
import termios
import threading
import time
import tty
import torch

# ===== ROS2 Lib =====
from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# ===== Component Lib =====
from omx_controller.models.BC.bc_model import BCPolicy

class NewController(Node):

    def __init__(self):
        super().__init__('keyboard_controller')

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

        self.joint_received = False

        self.max_delta = 0.02
        self.gripper_delta = 0.1
        self.last_command_time = time.time()
        self.command_interval = 0.02

        self.running = True  # for thread loop control

        self.get_logger().info('Waiting for /joint_states...')
        self.rate = self.create_rate(20)  # 20 Hz, equivalent to 0.05s timer

        # Logging throttle variables (log every 1 second for each type)
        self.last_joint_log_time = 0.0
        self.last_arm_log_time = 0.0
        self.last_gripper_log_time = 0.0
        self.log_interval = 1.0  # Adjust this to control logging speed (in seconds)

        # ===== Model ======
        self.model = BCPolicy(state_dim=6, action_dim=6)
        self.model.load_state_dict(torch.load("src/omx_controller/omx_controller/models/BC/bc_model.pth", weights_only=True))
        self.model.eval()

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

    def send_arm_command(self):
        arm_msg = JointTrajectory()
        arm_msg.joint_names = self.arm_joint_names
        arm_point = JointTrajectoryPoint()
        arm_point.positions = self.arm_joint_positions
        arm_point.time_from_start.sec = 0
        arm_msg.points.append(arm_point)
        self.arm_publisher.publish(arm_msg)

        # Throttled logging
        current_time = time.time()
        if current_time - self.last_arm_log_time >= self.log_interval:
            self.get_logger().info(f'Arm command sent: {self.arm_joint_positions}')
            self.last_arm_log_time = current_time

    def send_gripper_command(self):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = self.gripper_position
        goal_msg.command.max_effort = 10.0

        # Throttled logging
        current_time = time.time()
        if current_time - self.last_gripper_log_time >= self.log_interval:
            self.get_logger().info(f'Sending gripper command: {goal_msg.command.position}')
            self.last_gripper_log_time = current_time

        if not self.gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().error('Gripper action server not available')
            return
        send_goal_future = self.gripper_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected')
            return
        self.get_logger().info('Gripper goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.gripper_get_result_callback)

    def gripper_get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Gripper result: {result.reached_goal}')

    def run(self):
        while not self.joint_received and rclpy.ok() and self.running:
            self.get_logger().info('Waiting for initial joint states...')
            time.sleep(1.0)  # No spin_once needed here since spin is in main

        self.get_logger().info('Ready to run model control!')

        try:
            while rclpy.ok() and self.running:
                if not self.joint_received:
                    continue

                state = self.arm_joint_positions + [self.gripper_position]

                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    action_t = self.model.act(state_t, deterministic=True)

                moves = action_t.squeeze(0).cpu().numpy()
                self.arm_joint_positions = list(moves[:5])
                self.gripper_position = float(moves[5])
                self.send_arm_command()
                self.send_gripper_command()

                self.rate.sleep()

        except Exception as e:
            self.get_logger().error(f'Exception in run loop: {e}')

def main():
    rclpy.init()
    node = NewController()

    control_thread = threading.Thread(target=node.run)
    control_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Shutting down...')
    finally:
        node.running = False
        control_thread.join()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()