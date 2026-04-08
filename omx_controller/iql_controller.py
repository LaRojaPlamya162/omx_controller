#!/usr/bin/env python3

# ===== System Lib =====
import sys
import time
import torch
import csv
import os
import numpy as np
from pathlib import Path 

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
from omx_controller.models.IQL.IQL_agent import IQL
from omx_controller.components.reward import RewardFunction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "src/omx_controller/omx_controller/models/IQL"
#LOG_PATH = "src/omx_controller/omx_controller/models/SAC/logs_2"
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
        self.gripper_send_interval = 0.05

        # Logging throttle (every 1 second)
        self.last_joint_log_time = 0.0
        self.last_arm_log_time = 0.0
        self.last_gripper_log_time = 0.0
        self.last_ball_log_time = 0.0
        self.log_interval = 1.0

        # ===== IQL =====
        iql = IQL(state_dim=9, action_dim=6, device=DEVICE)
        iql.load_checkpoint(os.path.join(MODEL_DIR, "iql_epoch_10.pth"))
        iql.pi.eval()
        
    

        # Control / Logging variables
        self.prev_arm_positions = None
        self.prev_gripper_position = None
        self.prev_ball_pos = None
        self.prev_wrist_pos = None
        self.prev_state = None
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
        self.path = self.create_log_file()
        self.csv_file = open(self.path, "w", newline="") #open("src/omx_controller/omx_controller/models/SAC/sac_log.csv", "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            'episode','timestep',

            # state_t
            's1','s2','s3','s4','s5','g_s','rb_x','rb_y','rb_z',

            # action
            'a1','a2','a3','a4','a5','g_a',

            # next_state
            'ns1','ns2','ns3','ns4','ns5','ng_s','nrb_x','nrb_y','nrb_z',

            # ee position
            'jp_x','jp_y','jp_z',

            # ball position
            'bp_x','bp_y','bp_z',

            # distance
            'distance',

            'reward','done'
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
        #print("joint callback")

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
        if self.timestep >= 20000:
            exit()
        if not self.joint_received:
            return

        if self.initial_omx_pose is None:
            self.initial_omx_pose = self.arm_joint_positions + [self.gripper_position]
            self.get_logger().info("Initial OMX pose captured!")

        # ================= RESET STATE MACHINE =================
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

        # ================= TF END EFFECTOR =================
        try:
            transform = self.tf_buffer.lookup_transform(
                'world',
                'end_effector_link',
                rclpy.time.Time()
            )

            ee_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]

        except Exception:
            return

        self.joint_pos = ee_pos
        
        distance = np.linalg.norm(np.array(self.ball_pos) - np.array(ee_pos))

        # ================= BUILD CURRENT STATE (9D) =================
        relative_ball = (
            np.array(self.ball_pos) - np.array(self.joint_pos)
        ).tolist()

        current_state = (
            self.arm_joint_positions 
            + [self.gripper_position] 
            + relative_ball
        )

        # ================= REWARD =================
    
        if self.prev_ball_pos is not None:
            reward_fn = RewardFunction(
                self.ball_pos,
                self.joint_pos,
                self.prev_ball_pos,
                self.prev_wrist_pos,
                self.episode_step
            )

            reward = reward_fn.reward
            done = reward_fn.done

        else:
            reward = 0.0
            done = False

        # ================= PUSH REPLAY =================
        if self.prev_state is not None and self.prev_action is not None:

            # CSV log (s_t, a_t, s_t+1)
            row = (
                [self.episode] +
                [self.timestep] +
                self.prev_state +
                self.prev_action +
                current_state +
                self.joint_pos +
                self.ball_pos +
                [distance] +
                [reward] +
                [done]
            )

            self.writer.writerow(row)

            if self.timestep % 50 == 0:
                self.csv_file.flush()
            self.timestep += 1
            self.episode_step += 1
            self.prev_wrist_pos = self.joint_pos.copy()
            self.prev_ball_pos = self.ball_pos.copy()
            # Episode end
            if done:
                self.get_logger().info("Episode done -> start reset")
                self.reset_state = 'reset_robot'
                self.resetting = True
                self.new_episode_ready = False
                self.episode += 1
                self.episode_step = 0

                self.prev_state = None
                self.prev_action = None
                self.prev_ball_pos = None
                self.prev_wrist_pos = None
                return

        state_tensor = torch.tensor(
            current_state,
            dtype=torch.float32,
            device=DEVICE
        )

        state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-6)

        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action_tensor = self.bc_model.act(state_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()

        action = np.clip(action, -1.0, 1.0)


        # ===== SCALE DELTA =====
        max_delta = 0.05 # 0.3
        delta = action * max_delta
        current_joint = np.array(self.arm_joint_positions)
        arm_delta = action[:5] * 0.03
        gripper_delta = action[5] * 0.1
        joint_command = current_joint + arm_delta
        gripper_command = np.clip(
            self.gripper_position + gripper_delta,
            self.gripper_min,
            self.gripper_max
        )

        # Save action (policy output)
        self.prev_action = action.tolist()
        # Send command
        self.send_arm_command(joint_command.tolist())
        self.send_gripper_command(float(gripper_command))

        # ================= UPDATE PREV STATE =================
        self.prev_state = current_state

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
        self.reset_state = 'none'
        self.resetting = False
        self.new_episode_ready = True
        self.ball_reset_in_progress = False
    
    def create_log_file(self):
        log_dir = Path("src/omx_controller/omx_controller/models/BC/logs_1")
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
        return log_path

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