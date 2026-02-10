#!/usr/bin/env python3

# ===== System Lib =====
import select
import sys
import termios
import threading
import time
import tty
import torch
import csv

# ===== ROS2 Lib =====
from control_msgs.action import GripperCommand
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ros_gz_interfaces.srv import SetEntityPose
# ===== Component Lib =====
from omx_controller.models.BC.bc_model import BCPolicy

class Controller(Node):

    def __init__(self):
        super().__init__('keyboard_controller')
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
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
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/model/cricket_ball/pose',
            self.pose_callback,
            qos_profile
        )
        # Client to reset pose
        self.reset_client = self.create_client(SetEntityPose, '/world/default/set_pose')
        '''while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for set_pose service...')'''

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

        self.joint_received = False

        self.max_delta = 0.02
        self.gripper_delta = 0.1
        self.last_command_time = time.time()
        self.command_interval = 0.02

        self.running = True  # for thread loop control

        self.get_logger().info('Waiting for /joint_states...')
        self.rate = self.create_rate(20)  # 20 Hz, equivalent to 0.05s timer
        #self.timer = self.create_timer(0.05, self.control_step)

        # Time variables
        self.last_gripper_send_time = 0.0
        self.gripper_send_interval = 0.2
        
        # Logging throttle variables (log every 1 second for each type)
        self.last_joint_log_time = 0.0
        self.last_arm_log_time = 0.0
        self.last_gripper_log_time = 0.0
        self.log_interval = 1.0  # Adjust this to control logging speed (in seconds)

        # ===== Model ======
        self.model = BCPolicy(state_dim=6, action_dim=6)
        #self.model.load_state_dict(torch.load('omx_controller/models/BC/bc_model_v2.pth'))
        self.model.load_state_dict(torch.load("src/omx_controller/omx_controller/models/BC/bc_model.pth", weights_only=True))
        self.model.eval()

        # ===== Control / Logging variables =====
        self.timestep = 0
        self.prev_arm_positions = None
        self.prev_gripper_position = None

        # CSV logging
        self.csv_file = open("src/omx_controller/omx_controller/models/BC/bc_log.csv", "w", newline="")
        self.writer = csv.writer(self.csv_file)   # ← fixed
        self.writer.writerow([
            'episode',"timestep",
            "s1", "s2", "s3", "s4", "s5", "g_s",
            "n_s1", "n_s2", "n_s3", "n_s4", "n_s5", "n_g_s",
            "a1", "a2", "a3", "a4", "a5", "g_a"
        ])
    def joint_state_callback(self, msg):
        if set(self.arm_joint_names).issubset(set(msg.name)):
            for i, joint in enumerate(self.arm_joint_names):
                index = msg.name.index(joint)
                self.arm_joint_positions[i] = msg.position[index]

        if 'rh_r1_joint' in msg.name:
            index = msg.name.index('rh_r1_joint')
            self.gripper_position = msg.position[index]

        self.joint_received = True
        self.is_resetting = False
        # Throttled logging
        current_time = time.time()
        if current_time - self.last_joint_log_time >= self.log_interval:
            self.get_logger().info(
                f'Received joint states: {self.arm_joint_positions}, '
                f'Gripper: {self.gripper_position}'
            )
            self.last_joint_log_time = current_time
    def pose_callback(self, msg: PoseStamped):
        # Callback to handle received pose (e.g., log or process it)
        self.get_logger().info(f'Cricket ball pose: position={msg.pose.position}, orientation={msg.pose.orientation}')

    def reset_pose(self, x=0.0, y=0.0, z=1.0, roll=0.0, pitch=0.0, yaw=0.0):
        # Call service to reset pose
        req = SetEntityPose.Request()
        req.entity.name = 'cricket_ball'
        req.pose.position.x = x
        req.pose.position.y = y
        req.pose.position.z = z
        req.pose.orientation.x = roll
        req.pose.orientation.y = pitch
        req.pose.orientation.z = yaw
        req.pose.orientation.w = 1.0  # Assuming no rotation; adjust quaternion as needed

        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info('Pose reset successful')
        else:
            self.get_logger().error('Pose reset failed')
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
            current_time = time.time()

            if current_time - self.last_gripper_send_time < self.gripper_send_interval:
                return

            self.last_gripper_send_time = current_time

            goal_msg = GripperCommand.Goal()
            goal_msg.command.position = self.gripper_position
            goal_msg.command.max_effort = 10.0

            # Throttled logging
            if current_time - self.last_gripper_log_time >= self.log_interval:
                self.get_logger().info(f'Sending gripper command: {goal_msg.command.position:.4f}')
                self.last_gripper_log_time = current_time

            if not self.gripper_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn('Gripper action server not available')
                return

            send_goal_future = self.gripper_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(self.gripper_goal_response_callback)

    def gripper_goal_response_callback(self, future):
        """goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Gripper goal rejected')
            return
        self.get_logger().info('Gripper goal accepted')
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.gripper_get_result_callback)"""
        pass

    def gripper_get_result_callback(self, future):
        """result = future.result().result
        self.get_logger().info(f'Gripper result: {result.reached_goal}')"""

    def run(self):

        self.get_logger().info('Ready to run model control!')

        # Wait until first joint state received
        while rclpy.ok() and not self.joint_received:
            time.sleep(0.1)

        # Save initial pose (copy!)
        self.initial_arm_positions = self.arm_joint_positions.copy()
        self.initial_gripper_position = self.gripper_position

        self.prev_arm_positions = self.arm_joint_positions.copy()
        self.prev_gripper_position = self.gripper_position

        while rclpy.ok() and self.running:

            # ===== RESET LOGIC =====
            if self.timestep >= 2000 and not self.is_resetting:

                self.is_resetting = True
                self.get_logger().info("=== START RESET ===")

                # 1. Stop model loop
                self.running = False

                # 2. Reset ball
                self.reset_pose()

                # 3. Reset robot
                self.arm_joint_positions = self.initial_arm_positions.copy()
                self.gripper_position = self.initial_gripper_position

                self.send_arm_command()
                self.send_gripper_command()

                # 4. Wait robot stabilize
                time.sleep(2.0)

                # 5. Reset counters
                self.timestep = 0

                # 6. Resume loop
                self.running = True
                self.is_resetting = False

                self.get_logger().info("=== RESET DONE ===")
                continue
            # ===== DO NOT PUBLISH DURING RESET =====
            if self.is_resetting:
                continue

            # ===== MODEL CONTROL =====
            current_state = self.arm_joint_positions + [self.gripper_position]
            state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action_tensor = self.model.act(state_tensor, deterministic=True)

            action = action_tensor.squeeze(0).cpu().numpy()
            desired_arm = action[:5].tolist()
            desired_gripper = float(action[5])

            # ===== LOG CSV =====
            row = (
                [self.timestep] +
                self.prev_arm_positions +
                [self.prev_gripper_position] +
                self.arm_joint_positions +
                [self.gripper_position] +
                desired_arm +
                [desired_gripper]
            )
            self.writer.writerow(row)
            self.csv_file.flush()

            # ===== APPLY ACTION =====
            self.arm_joint_positions = desired_arm
            self.gripper_position = desired_gripper

            self.send_arm_command()
            self.send_gripper_command()

            self.prev_arm_positions = self.arm_joint_positions.copy()
            self.prev_gripper_position = self.gripper_position

            self.timestep += 1

            self.rate.sleep()


def main():
    rclpy.init()
    node = Controller()

    control_thread = threading.Thread(target=node.run)
    control_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\nCtrl+C detected. Shutting down...')
    finally:
        node.running = False
        control_thread.join()
        
        # Close CSV file
        if hasattr(node, 'csv_file'):
            node.csv_file.close()
            print("CSV log file closed.")
            
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()