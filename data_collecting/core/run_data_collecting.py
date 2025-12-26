import os
os.environ['QT_LOGGING_RULES'] = "qt.qpa.*=false"

from pathlib import Path
import time
import sys

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import glob
import cv2
import numpy as np
import tyro
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path

# --- Level 1 & 2 Imports (Custom Modules) ---
# Ensure python path sees the root directory
sys.path.append(str(Path(__file__).parent.parent))

from data_collecting.utils.logger import Logger as log
from data_collecting.utils.config_loader import ConfigLoader
from data_collecting.utils.saver import DataSaver
from data_collecting.devices.realsense import RealSenseCamera

# --- Gello Imports ---
from third_party.gello_software.gello.env import RobotEnv
from third_party.gello_software.gello.zmq_core.robot_node import ZMQClientRobot
from third_party.gello_software.gello.utils.launch_utils import instantiate_from_dict

# Load Configs globally
cfg_loader = ConfigLoader()
robot_cfg = cfg_loader.get_robot()
task_cfg = cfg_loader.get_task()
hardware_cfg = cfg_loader.hardware
cameras_cfg = hardware_cfg.get("cameras", {})

@dataclass
class Args:
    # Network / Robot
    robot_port: int = robot_cfg.get("port", 6001)
    hostname: str = "127.0.0.1" # The IP where arm_server.py is running
    
    # Gello Agent
    agent: str = "gello"
    gello_port: Optional[str] = None
    start_joints: Optional[Tuple[float, ...]] = None
    
    # Camera Settings
    front_camera_id: str = cameras_cfg.get("front", {}).get("device_id", "default")
    side_camera_id: str = cameras_cfg.get("side", {}).get("device_id", "default")
    
    # Collection Settings
    hz: int = 100
    data_dir: str = cfg_loader.get_storage_dir()
    task_name: str = task_cfg.get("name", "default_task")
    
    def __post_init__(self):
        if self.start_joints is not None:
            self.start_joints = np.array(self.start_joints)

def main(args: Args):
    # ================= 1. Initialization =================
    log.info(f"Initializing Data Collection for Task: {args.task_name}")
    
    # A. Setup Saver

    save_path = Path(args.data_dir).expanduser() / args.task_name
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        episode_idx = 0
    else:
        existing_episodes = list(save_path.glob("episode_*.h5"))
        if not existing_episodes:
            episode_idx = 0
        else:
            try:
                indices = [int(f.name.split('_')[1]) for f in existing_episodes]
                episode_idx = max(indices) + 1
            except (IndexError, ValueError):
                log.warn("Starting from episode idx 0.")
                episode_idx = 0
    
    log.info(f"Auto-resuming: Next Episode Index will be {episode_idx}")

    # Load task instructions from config (e.g. ["Pick apple", "Place apple"])
    task_instructions = task_cfg.get("instructions", ["Default Task"])
    stages_num = task_cfg.get("stages_num", 1)
    
    saver = DataSaver(
        save_dir=args.data_dir, 
        task_name=args.task_name, 
        description=task_cfg.get("description", "")
    )

    # B. Setup Camera (Level 2 Device) - Optional cameras
    # Initialize cameras sequentially with delay to avoid USB bandwidth conflicts
    front_cam = None
    side_cam = None
    
    # Initialize front camera only if device_id is specified and not "default"
    if args.front_camera_id and args.front_camera_id != "default":
        log.info(f"Initializing front camera with device_id: {args.front_camera_id}")
        front_cam = RealSenseCamera(device_id=args.front_camera_id, width=640, height=480, fps=30)
        if front_cam.pipeline is None:
            log.warn("Front camera initialization failed, continuing without it.")
            front_cam = None
        else:
            log.success("Front camera initialized successfully.")
            # Wait a bit to ensure first camera is fully stable before initializing second
            log.info("Waiting for front camera to stabilize before initializing side camera...")
            time.sleep(2.0)
    else:
        log.info("Front camera not configured, skipping initialization.")
    
    # Initialize side camera only if device_id is specified and not "default"
    if args.side_camera_id and args.side_camera_id != "default":
        log.info(f"Initializing side camera with device_id: {args.side_camera_id}")
        side_cam = RealSenseCamera(device_id=args.side_camera_id, width=640, height=480, fps=30)
        if side_cam.pipeline is None:
            log.warn("Side camera initialization failed, continuing without it.")
            side_cam = None
        else:
            log.success("Side camera initialized successfully.")
    else:
        log.info("Side camera not configured, skipping initialization.")
    
    # Check if at least one camera is available
    if front_cam is None and side_cam is None:
        log.error("No cameras available! Please configure at least one camera in hardware.yaml")
        return


    # C. Setup Robot Client (Connects to arm_server.py)
    log.info(f"Connecting to Robot Server at {args.hostname}:{args.robot_port}...")
    try:
        robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
        env = RobotEnv(robot_client, control_rate_hz=args.hz, camera_dict={})
        log.success("Robot Connected.")
    except Exception as e:
        log.error(f"Failed to connect to robot: {e}")
        return

    # D. Setup Agent (Gello Handle)
    agent_cfg = {}
    if args.agent == "gello":
        gello_port = args.gello_port
        if gello_port is None:
            # Auto-detect USB serial
            usb_ports = glob.glob("/dev/serial/by-id/*")
            if len(usb_ports) > 0:
                gello_port = usb_ports[0]
                log.info(f"Found Gello Port: {gello_port}")
            else:
                log.error("No Gello port found! Please check USB connection.")
                return
        
        agent_cfg = {
            "_target_": "third_party.gello_software.gello.agents.gello_agent.GelloAgent",
            "port": gello_port,
            "start_joints": args.start_joints,
        }
    elif args.agent == "dummy":
        agent_cfg = {"_target_": "third_party.gello_software.gello.agents.agent.DummyAgent", "num_dofs": 7}
    
    # Instantiate the agent
    try:
        agent = instantiate_from_dict(agent_cfg)
    except Exception as e:
        log.error(f"Failed to instantiate agent: {e}")
        return

    # ================= 2. Synchronization (Safety) =================
    log.info(">>> Please align Gello handle with Robot position <<<")
    
    last_print_time = 0
    while True:
        obs = env.get_obs()
        joints = obs["joint_positions"]
        agent_cmd = agent.act(obs)
        
        abs_deltas = np.abs(agent_cmd - joints)
        max_diff = np.max(abs_deltas)
        
        if max_diff < 0.8: # Threshold
            log.success("Sync Successful! Control active.")
            break
        
        if time.time() - last_print_time > 0.5:
            log.info(f"Syncing... Max Diff: {max_diff:.3f} rad")
            last_print_time = time.time()
        time.sleep(0.01)

    # Soft Start (Smooth interpolation to target)
    log.info("Soft starting...")
    for _ in range(50):
        obs = env.get_obs()
        target = agent.act(obs)
        curr = obs["joint_positions"]
        delta = target - curr
        scale = 0.05 / max(np.max(np.abs(delta)), 0.001)
        scale = min(scale, 1.0)
        env.step(curr + delta * scale)
        time.sleep(0.01)

    # ================= 3. Main Collection Loop =================
    log.info("Starting Main Loop. Controls:")
    log.info("  [R] Start/Stop Recording")
    log.info("  [T] Next Task Stage")
    log.info("  [Q] Quit")

    # State Variables
    recording = False
    current_stage_idx = 0
    
    # Data Buffers
    buffer = {} 
    task_switch_frames = [] # Stores [stage_idx, frame_idx]

    try:
        while True:
            t_start = time.time()

            # --- A. Read Sensors ---
            # Read from available cameras
            color_front, depth_front = (None, None)
            color_side, depth_side = (None, None)
            
            if front_cam is not None:
                color_front, depth_front = front_cam.read()
            if side_cam is not None:
                color_side, depth_side = side_cam.read()
            
            obs = env.get_obs()
            
            # --- B. Get Action & Step ---
            action = agent.act(obs)
            env.step(action)
            
            # --- C. Recording Logic ---
            # Record if at least one camera is available and recording
            has_camera_data = (color_front is not None) or (color_side is not None)
            if recording and has_camera_data:
                # Store Data
                # Note: We resize images to save space (Optional, adjust fx/fy as needed)
                small_c_front = None
                small_d_front = None
                small_c_side = None
                small_d_side = None
                
                if color_front is not None:
                    small_c_front = cv2.resize(color_front, (0,0), fx=0.5, fy=0.5)
                    small_d_front = cv2.resize(depth_front, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                
                if color_side is not None:
                    # Side camera: rotate first, then resize
                    rotated_side = cv2.rotate(color_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    small_c_side = cv2.resize(rotated_side, (0,0), fx=0.5, fy=0.5)
                    rotated_depth_side = cv2.rotate(depth_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    small_d_side = cv2.resize(rotated_depth_side, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
                
                # Concatenate images for storage if both cameras are available
                color_concate = None
                if small_c_front is not None and small_c_side is not None:
                    # Resize both to same height for concatenation
                    h_front, w_front = small_c_front.shape[:2]
                    h_side, w_side = small_c_side.shape[:2]
                    
                    # Use the smaller height as target
                    target_h = min(h_front, h_side)
                    # Resize to same height while maintaining aspect ratio
                    scale_front = target_h / h_front
                    scale_side = target_h / h_side
                    new_w_front = int(w_front * scale_front)
                    new_w_side = int(w_side * scale_side)
                    
                    resized_front = cv2.resize(small_c_front, (new_w_front, target_h))
                    resized_side = cv2.resize(small_c_side, (new_w_side, target_h))
                    color_concate = np.hstack([resized_front, resized_side])
                elif small_c_front is not None:
                    color_concate = small_c_front.copy()
                elif small_c_side is not None:
                    color_concate = small_c_side.copy()
                
                if 'colors_front' not in buffer: 
                    # Initialize buffer if empty
                    buffer = {
                        'colors_front': [], 'colors_side': [], 'colors_concate': [],
                        'depths_front': [], 'depths_side': [],
                        'joints': [], 'ee_poses': [], 'grippers': [], 'timestamps': [], 'actions': []
                    }

                if small_c_front is not None:
                    buffer['colors_front'].append(small_c_front)
                    buffer['depths_front'].append(small_d_front)
                if small_c_side is not None:
                    buffer['colors_side'].append(small_c_side)
                    buffer['depths_side'].append(small_d_side)
                if color_concate is not None:
                    buffer['colors_concate'].append(color_concate)
                
                buffer['joints'].append(obs['joint_positions'])
                buffer['ee_poses'].append(obs.get('ee_pos_quat', np.zeros(7))) # Ensure key exists
                buffer['grippers'].append(obs.get('gripper_position', [0])[0])
                buffer['actions'].append(action)
                buffer['timestamps'].append(time.time())

            # --- D. Visualization ---
            # Display available camera(s)
            if color_front is not None or color_side is not None:
                disp = None
                
                if color_front is not None and color_side is not None:
                    # Both cameras available: concatenate
                    rotated_side = cv2.rotate(color_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Align sizes for display
                    h_front, w_front = color_front.shape[:2]
                    h_side, w_side = rotated_side.shape[:2]
                    target_h = min(h_front, h_side)
                    scale_front = target_h / h_front
                    scale_side = target_h / h_side
                    new_w_front = int(w_front * scale_front)
                    new_w_side = int(w_side * scale_side)
                    
                    resized_front = cv2.resize(color_front, (new_w_front, target_h))
                    resized_side = cv2.resize(rotated_side, (new_w_side, target_h))
                    
                    # Concatenate for display
                    disp = np.hstack([resized_front, resized_side])
                    
                    # Add labels for each camera view
                    cv2.putText(disp, "Front", (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.putText(disp, "Side", (new_w_front + 10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                elif color_front is not None:
                    # Only front camera
                    disp = color_front.copy()
                    cv2.putText(disp, "Front", (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                elif color_side is not None:
                    # Only side camera
                    disp = cv2.rotate(color_side, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.putText(disp, "Side", (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if disp is not None:
                    # UI: Recording Status
                    if recording:
                        cv2.circle(disp, (30, 30), 10, (0, 0, 255), -1) # Red Dot
                        frame_count = len(buffer.get('colors_concate', []))
                        cv2.putText(disp, f"REC {frame_count}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # UI: Current Task Stage
                        curr_instr = task_instructions[current_stage_idx] if current_stage_idx < len(task_instructions) else "Done"
                        cv2.putText(disp, f"Stage {current_stage_idx+1}: {curr_instr}", (20, disp.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        cv2.putText(disp, "STANDBY", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(disp, f"Ep: {episode_idx}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    cv2.imshow("Data Collection", disp)

            # --- E. Input Handling ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'): 
                break
                
            elif key == ord('r'): # Toggle Recording
                recording = not recording
                if recording:
                    log.info(f"Start Recording Episode {episode_idx}...")
                    # Reset buffers
                    buffer = {}
                    current_stage_idx = 0
                    # First stage always starts at frame 0
                    task_switch_frames = [[0, 0]] 
                else:
                    log.info(f"Stop Recording. Saving Episode {episode_idx}...")
                    # Call our robust Saver
                    success = saver.save_episode(
                        buffer, 
                        episode_idx, 
                        task_descriptions=task_instructions,
                        task_switch_frames=task_switch_frames,
                        expected_stages=stages_num
                    )
                    if success:
                        episode_idx += 1
            
            elif key == ord('t') and recording: # Toggle Task Stage
                if current_stage_idx < len(task_instructions) - 1:
                    current_stage_idx += 1
                    frame_idx = len(buffer.get('colors_front', []))
                    task_switch_frames.append([current_stage_idx, frame_idx])
                    log.info(f" -> Switched to Stage {current_stage_idx+1}: {task_instructions[current_stage_idx]}")
                else:
                    log.warn("Already at last stage.")

            # Loop Rate Control
            dt = time.time() - t_start
            if dt < 1/args.hz: 
                time.sleep(1/args.hz - dt)

    except KeyboardInterrupt:
        log.info("Keyboard Interrupt.")
    except Exception as e:
        log.error(f"Runtime Error: {e}")
    finally:
        if front_cam is not None:
            front_cam.stop()
        if side_cam is not None:
            side_cam.stop()
        cv2.destroyAllWindows()
        log.info("Program Exited.")

if __name__ == "__main__":
    main(tyro.cli(Args))