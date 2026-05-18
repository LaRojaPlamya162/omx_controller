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
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import PoseArray

# ===== Component Lib =====
from omx_controller.models.SAC.sac_model import SACAgent, initialize_sac_from_bc
from omx_controller.components.reward import RewardFunction
from omx_controller.models.SAC.replay_buffer import ReplayBuffer, fill_replay_buffer_from_dataframe
from omx_controller.components.utils import get_action_max_min, dataset_length, create_log_file, concat_dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "src/omx_controller/omx_controller/models/BC_to_SAC"
BC_DIR = "src/omx_controller/omx_controller/models/BC"
#LOG_PATH = "src/omx_controller/omx_controller/models/SAC/logs_2"
class Controller(Node):

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

        self.ball_sub = self.create_subscription(
                    PoseArray,
                    #TFMessage,
                    '/world/empty/pose/info',
                    self.ball_callback,
                    10
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
        self.gripper_max = 1.0
        self.gripper_min = 0.0
        #self.initial_ball_pose = [0.0, 2.0, 1.0]  # Consistent with spawn position
        self.joint_received = False
        self.initial_omx_pose = None
        self.ball_pos = [0.2, 0.2, 0.0]  # Default ball position
        self.joint_pos = None
        self.pose_index = None
        # Control parameters
        self.last_command_time = time.time()
        self.command_interval = 0.02

        # Gripper send throttling
        self.last_gripper_send_time = 0.0
        self.gripper_send_interval = 0.05

        # Logging throttle (every 1 second)
        self.last_joint_log_time = 0.0
        self.last_arm_log_time = 0.0
        self.log_interval = 1.0

        # Reward
        self.ball_in_target_steps = 0
        self.min_steps_in_target = 50
        self.ball_out_of_playground_steps = 0
        self.done = False
        # ===== SAC =====
        action_max, action_min = get_action_max_min()
        self.action_min = action_min.cpu().numpy()
        self.action_max = action_max.cpu().numpy()
        self.agent = SACAgent(state_dim = 9, action_dim = 6)
        bc_checkpoint_path = os.path.join(BC_DIR,"bc_model_v2_squashed_real.pth")
        if os.path.exists(os.path.join(MODEL_DIR, "checkpoint_1/SAC.pth")):
            # Đã từng fine-tune rồi → load checkpoint (đã có weights tốt)
            self.agent.load_checkpoint(os.path.join(MODEL_DIR, "checkpoint_1/SAC.pth"))
            self.get_logger().info(f"✅ Loaded fine-tuned SAC checkpoint_1")
        else:
            self.agent = initialize_sac_from_bc(
                self.agent, 
                bc_checkpoint_path, 
                state_dim=9, 
                action_dim=6
            )
            self.get_logger().info(f"✅ Initialized SAC Actor from BC policy")
        
        # ===== Load replay buffer =====
        self.replay = ReplayBuffer(state_dim = 9, action_dim = 6, capacity = 1000000, device = DEVICE)
        if os.path.exists(os.path.join(MODEL_DIR,"checkpoint_1/replay.pth")):
            self.replay.load(os.path.join(MODEL_DIR, "checkpoint_1/replay.pth"))
        else:
            # If not exist -> load from BC
            required_fields = [
                's1','s2','s3','s4','s5','g_s','rb_x','rb_y','rb_z',
                'a1','a2','a3','a4','a5','g_a',
                'ns1','ns2','ns3','ns4','ns5','ng_s','nrb_x','nrb_y','nrb_z',
                'reward','done'
            ]
            
            bc_files = [os.path.join(BC_DIR, f"logs_3/log_{i}.csv") for i in range(1, 4)]
            self.get_logger().info(f"Đang load {len(bc_files)} file BC để warm-up replay buffer...")
            df = concat_dataset(files=bc_files, col_names=required_fields)   # hàm của bạn
            fill_replay_buffer_from_dataframe(self.replay, df, verbose=True)
            self.get_logger().info(f"✅ Filled replay buffer from BC data: {len(self.replay):,} transitions")
        if os.path.exists(os.path.join(MODEL_DIR, "logs_1")):
            self.training_size = dataset_length(os.path.join(MODEL_DIR, "logs_1"))
        else:
            self.training_size = 0
        self.get_logger().info(f"Model has been training for {len(self.replay)} timesteps!")
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

        # real world stats
        checkpoint = torch.load(bc_checkpoint_path)
        self.state_mean = checkpoint['state_mean'].to(DEVICE)
        self.state_std = checkpoint['state_std'].to(DEVICE)
        # CSV logging
        self.path, self.index = create_log_file(os.path.join(MODEL_DIR, "logs_1"))
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
            'distance','reward',

            # status
            'ball_in_target','ball_out_of_playground', 'time_limit', 'done'
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
                f'Timestep: {self.timestep}, '
                f'Received joint states: {self.arm_joint_positions}, '
                f'Gripper: {self.gripper_position}'
            )
            self.last_joint_log_time = current_time

    def ball_callback(self, msg):
        if self.pose_index is None:
            for i, pose in enumerate(msg.poses):
                if abs(pose.position.x - 0.2) < 0.01 and abs(pose.position.y - 0.2) < 0.01:
                #if pose.position.x == 0.2 and pose.position.y == 0.2:
                    self.pose_index = i
                    self.get_logger().info(f"Episode: {self.episode}, Ball pose index: {self.pose_index}")
        else:
            ball_pose_msg = msg.poses[self.pose_index] 
                    
            x = float(ball_pose_msg.position.x)
            y = float(ball_pose_msg.position.y)
            z = float(0.0) if ball_pose_msg.position.z < 0.0 else float(ball_pose_msg.position.z)

            self.ball_pos = [x, y, z]
            #self.get_logger().info(f"Timestep {self.timestep}, Ball pos: {self.ball_pos}")

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
            self.get_logger().info(f'Timestep: {self.timestep}, Arm command sent: {arm_pos}')
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
        if self.timestep >= 10000 or self.episode >= 25:
            self.get_logger().info(f"Log completed")
            exit()
        if len(self.replay) >= 120000:
            self.get_logger().info(f"Training completed")
            exit()
        if not self.joint_received:
            return
        
        if self.initial_omx_pose is None:
            self.initial_omx_pose = self.arm_joint_positions + [self.gripper_position]
            self.get_logger().info("Initial OMX pose captured!")

        #self.get_logger().info(f"Timestep {self.timestep}, ball in the bounding box for {self.ball_in_target_steps} steps, ball out of playground for {self.ball_out_of_playground_steps} steps and done: {self.done}")
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
            if reward_fn.is_success():
                self.ball_in_target_steps +=1
            else:
                self.ball_in_target_steps = 0
            if not reward_fn.is_ball_in_playground():
                self.ball_out_of_playground_steps += 1
            else:
                self.ball_out_of_playground_steps = 0

            self.done = (self.ball_in_target_steps >= self.min_steps_in_target) or reward_fn.check_out_of_time() or (self.ball_out_of_playground_steps >= 5)

        else:
            reward = 0.0
            self.done = False
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
                [self.ball_in_target_steps] +
                [self.ball_out_of_playground_steps] +
                [self.episode_step] +
                [self.done]
            )

            self.writer.writerow(row)

            if self.timestep % 50 == 0:
                self.csv_file.flush()

            self.replay.push(
                self.prev_state,
                self.prev_action,
                reward,
                current_state,
                self.done
            )
            self.timestep += 1
            self.episode_step += 1
            self.prev_wrist_pos = self.joint_pos.copy()
            self.prev_ball_pos = self.ball_pos.copy()
            
        # Episode end
        if self.done:
                self.get_logger().info("Episode done -> start reset")
                if len(self.replay) > 0 and self.timestep > 0:
                    self.agent.save_checkpoint(
                        os.path.join(MODEL_DIR, "checkpoint_1/SAC.pth")
                    )
                    self.replay.save(os.path.join(MODEL_DIR, "checkpoint_1/replay.pth"))
                    self.get_logger().info("Save SAC model and replay buffer")
                self.reset_state = 'reset_robot'
                self.resetting = True
                self.new_episode_ready = False
                self.episode += 1
                self.episode_step = 0
                self.prev_state = None
                self.prev_action = None
                self.prev_ball_pos = None
                self.prev_wrist_pos = None
                self.pose_index = None
                self.ball_in_target_steps = 0
                self.ball_out_of_playground_steps = 0
                return


        
        # ====== INFERENCE ======
        state_tensor = torch.tensor(
            current_state,
            dtype=torch.float32,
            device=DEVICE
        )

        state_tensor = (state_tensor - self.state_mean) / (self.state_std + 1e-6)

        state_tensor = state_tensor.unsqueeze(0)
        with torch.no_grad():
            action_tensor, _, _ = self.agent.actor.sample(state_tensor)
        action = action_tensor.squeeze(0).cpu().numpy()

        action = np.clip(action, -1.0, 1.0)
        
        # ===== Un-normalize =====
        real_action = self.action_min + (action + 1.0) * 0.5 * (self.action_max - self.action_min)
        joint_command = real_action[:5]
        gripper_command = real_action[5]
        gripper_command = np.clip(
            gripper_command,
            self.gripper_min,
            self.gripper_max
        )
        self.send_arm_command(joint_command.tolist())
        self.send_gripper_command(float(gripper_command))

        # Save action (policy output)
        self.prev_action = action.tolist()

        # ================= UPDATE PREV STATE =================
        self.prev_state = current_state

        # ================= TRAIN =================
        if len(self.replay) > 10000 and self.timestep % 10 == 0:
            for _ in range(5):
                self.agent.update(self.replay)
            
        if self.timestep % 100 == 0 and len(self.replay) > 0 and self.timestep > 0:
            self.agent.save_checkpoint(
                os.path.join(MODEL_DIR, "checkpoint_1/SAC.pth")
            )
            self.replay.save(os.path.join(MODEL_DIR, "checkpoint_1/replay.pth"))
            self.get_logger().info("Save SAC model and replay buffer")

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

        spawn_req.entity_factory.pose.position.x = float(0.2)
        spawn_req.entity_factory.pose.position.y = float(0.2)
        spawn_req.entity_factory.pose.position.z = float(0.0)

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
        self.done = True
        self.ball_in_target_steps = 0
        self.ball_out_of_playground_steps = 0

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

    return
if __name__ == '__main__':
    main()