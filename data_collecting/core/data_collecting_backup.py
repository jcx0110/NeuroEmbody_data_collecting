import time
import os
import glob
import h5py
import cv2
import numpy as np
import tyro
import pyrealsense2 as rs
from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime

# GELLO Library Imports
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.utils.launch_utils import instantiate_from_dict

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.align = rs.align(rs.stream.color)

        try:
            self.profile = self.pipeline.start(config)
            print(f"RealSense Camera Started ({width}x{height})")
            
            print("Warming up camera...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            print("Camera warmed up.")
            
        except Exception as e:
            print(f"Failed to start RealSense: {e}")
            self.pipeline = None

    def read(self):
        if not self.pipeline: return None, None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=2000) # Slightly increase timeout to prevent occasional frame drops
            
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                print("Warning: Empty frame received")
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            return color_image, depth_image
            
        except RuntimeError as e:
            print(f"RealSense Runtime Error: {e}")
            return None, None
        except Exception as e:
            print(f"Camera Read Error: {e}")
            return None, None

    def stop(self):
        if self.pipeline: 
            self.pipeline.stop()
            print("RealSense stopped.")

# ==========================================
# 2. Parameter Configuration
# ==========================================
@dataclass
class Args:
    agent: str = "gello" 
    robot_port: int = 6001
    hostname: str = "127.0.0.1"
    robot_type: str = None
    hz: int = 100
    start_joints: Optional[Tuple[float, ...]] = None
    gello_port: Optional[str] = None
    mock: bool = False
    use_save_interface: bool = False
    data_dir: str = "~/bc_data"
    bimanual: bool = False
    verbose: bool = False
    task_name: str = "test_task"

    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)

# ==========================================
# 3. Custom Save Function
# ==========================================
def save_data(save_dir, episode_idx, buffer, task_name):
    if not buffer['colors']: return
    path = os.path.join(save_dir, f"episode_{episode_idx:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("color", data=np.array(buffer['colors']), compression="gzip")
        f.create_dataset("depth", data=np.array(buffer['depths']), compression="gzip")
        f.create_dataset("joint_positions", data=np.array(buffer['joints']))
        f.create_dataset("ee_pose", data=np.array(buffer['ee_poses']))
        f.create_dataset("actions", data=np.array(buffer['actions']))
        f.attrs["task_name"] = task_name
    print(f"Saved {len(buffer['colors'])} frames to {path}")

# ==========================================
# 4. Main Program
# ==========================================
def main(args: Args):
    # --- A. Robot Connection ---
    print(f"Connecting to Robot at {args.hostname}:{args.robot_port}...")
    try:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict={})
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        return

    # --- B. Agent Initialization ---
    agent_cfg = {}
    if args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                print(f"Found Gello Port: {gello_port}")
            else:
                print("No Gello port found! Please plug it in.")
                return
        
        agent_cfg = {
            "_target_": "gello.agents.gello_agent.GelloAgent",
            "port": gello_port,
            "start_joints": args.start_joints,
        }
        
        # Robot reset logic (Reset to Home)
        if args.start_joints is None:
            reset_joints = np.array([-1.57, -1.57, -1.57, -1.57, 1.57, 3.14, 0.0])
        else:
            reset_joints = np.array(args.start_joints)

        curr_joints = env.get_obs()["joint_positions"]
        if reset_joints.shape == curr_joints.shape:
            print("Resetting robot to home position...")
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 100)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)
                time.sleep(0.001)

    elif args.agent == "dummy":
        agent_cfg = {"_target_": "gello.agents.agent.DummyAgent", "num_dofs": 7}
    else:
        print(f"Agent {args.agent} not supported in this script.")
        return

    # Instantiate Agent
    agent = instantiate_from_dict(agent_cfg)

    # --- C. Smart Sync [Modified Part] ---
    print("\nPerforming Safety Sync...")
    print("Please move the Gello handle to match the robot's position.")
    
    last_print_time = 0
    
    while True:
        obs = env.get_obs()
        joints = obs["joint_positions"] # Robot current position (Follower)
        agent_cmd = agent.act(obs)      # Handle current position (Leader)
        
        # Calculate deviation
        abs_deltas = np.abs(agent_cmd - joints)
        max_diff = np.max(abs_deltas)
        
        # Threshold set to 0.8 (consistent with official)
        if max_diff < 0.8:
            print("\nSync successful! Starting control...")
            break
        else:
            # Print detailed info every 0.5 seconds to avoid spamming
            if time.time() - last_print_time > 0.5:
                print("\n" + "-"*60)
                print(f"Syncing... Max Diff: {max_diff:.3f} rad")
                
                # Find all joints with deviation > 0.8
                error_indices = np.where(abs_deltas > 0.8)[0]
                
                for idx in error_indices:
                    delta = abs_deltas[idx]
                    leader_val = agent_cmd[idx]
                    follower_val = joints[idx]
                    # Output exactly in the requested format
                    print(f"joint[{idx}]: \t delta: {delta:4.3f} , leader: \t{leader_val:4.3f} , follower: \t{follower_val:4.3f}")
                
                print("-"*60)
                last_print_time = time.time()
            
            time.sleep(0.01)

    # Soft Start
    for _ in range(50):
        obs = env.get_obs()
        target = agent.act(obs)
        curr = obs["joint_positions"]
        delta = target - curr
        scale = 1.0
        if np.max(np.abs(delta)) > 0.05:
            scale = 0.05 / np.max(np.abs(delta))
        env.step(curr + delta * scale)
        time.sleep(0.01)

    # --- D. Data Collection Main Loop ---
    print("\nStarting Custom Collection Loop.")
    
    camera = RealSenseCamera()
    full_save_dir = os.path.expanduser(os.path.join(args.data_dir, args.task_name))
    os.makedirs(full_save_dir, exist_ok=True)
    
    buffer = {'colors': [], 'depths': [], 'joints': [], 'ee_poses': [], 'actions': []}
    recording = False
    episode_idx = 0

    print("==========================================")
    print("   [R] Record/Stop   [Q] Quit")
    print("==========================================")

    try:
        while True:
            t_start = time.time()
            
            # 1. Read sensors
            color, depth = camera.read()
            obs = env.get_obs()
            
            # 2. Read handle action
            action = agent.act(obs)
            
            # 3. Send to robot
            env.step(action)
            
            # 4. Recording logic
            if recording and color is not None:
                small_c = cv2.resize(color, (0,0), fx=0.5, fy=0.5)
                small_d = cv2.resize(depth, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                buffer['colors'].append(small_c)
                buffer['depths'].append(small_d)
                buffer['joints'].append(obs['joint_positions'])
                buffer['ee_poses'].append(obs['ee_pose'])
                buffer['actions'].append(action)

            # 5. Visualization
            if color is not None:
                disp = color.copy()
                status = f"REC {len(buffer['colors'])}" if recording else "STANDBY"
                col = (0,0,255) if recording else (0,255,0)
                cv2.putText(disp, status, (20,40), 4, 1, col, 2)
                cv2.imshow("Data Collection", disp)

            # 6. Key interaction
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                recording = not recording
                if recording:
                    print(f"Start Ep {episode_idx}")
                    buffer = {'colors': [], 'depths': [], 'joints': [], 'ee_poses': [], 'actions': []}
                else:
                    save_data(full_save_dir, episode_idx, buffer, args.task_name)
                    episode_idx += 1

            dt = time.time() - t_start
            if dt < 1/args.hz: time.sleep(1/args.hz - dt)

    except KeyboardInterrupt: pass
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main(tyro.cli(Args))