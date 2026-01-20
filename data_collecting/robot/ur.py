import sys
import numpy as np
from typing import Dict
from pathlib import Path

# Path handling to ensure logger and other utils can be imported
try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    # Allow imports if running from root or sub-folder
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log

# Import base class from third_party (Ensure Gello is in your python path)
try:
    from data_collecting.robot.robot import Robot
except ImportError:
    log.error("Gello software not found in third_party. Please check your path.")
    # Define a dummy Robot class to prevent immediate crash during syntax check
    class Robot: pass

class URRobot(Robot):
    def __init__(self, robot_ip: str = "192.168.1.2", no_gripper: bool = False, gripper_ip: str = "192.168.1.11"):
        """
        Initialize the UR Robot connection via RTDE.
        """
        import rtde_control
        import rtde_receive

        # 1. Initialize Robot Control Interface
        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
            log.success(f"Connected to UR Robot Control at {robot_ip}")
        except Exception as e:
            log.error(f"UR Connection Error: {e}")
            raise e

        # 2. Initialize Data Receive Interface
        try:
            self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        except Exception as e:
            log.error(f"UR Receive Interface Error: {e}")

        # 3. Initialize Gripper
        self._use_gripper = not no_gripper
        if self._use_gripper:
            try:
                # Relative import: assumes gripper.py is in data_collecting/robot/
                from .gripper import MyGripper
                
                log.info("Connecting to Custom Gripper...")
                # Note: You might want to parameterize this IP in hardware.yaml later
                self.gripper = MyGripper(ip_address=gripper_ip) 
                log.success("Gripper connected.")
            except ImportError:
                log.error("Could not import 'MyGripper'. Is 'gripper.py' in the same folder?")
                self._use_gripper = False
            except Exception as e:
                log.error(f"Gripper connection failed: {e}")
                self._use_gripper = False

        # 4. Disable Freedrive on startup for safety
        self._free_drive = False
        self.robot.endFreedriveMode()

    def num_dofs(self) -> int:
        """Return number of degrees of freedom (6 arm + 1 gripper)."""
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        """Get current joint positions (including gripper if enabled)."""
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            # Gripper state is usually appended as the last element
            # Note: You need to ensure gripper.get_current_position() is implemented if needed
            # For now, appending 0.0 or reading from gripper class
            pos = np.append(robot_joints, 0.0) 
        else:
            pos = robot_joints
        return np.array(pos)

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        """
        Send joint commands to the robot using ServoJ.
        :param joint_state: 7D array (6 joints + 1 gripper)
        """
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500  # 500Hz control loop
        lookahead_time = 0.2
        gain = 100

        # Split arm and gripper commands
        robot_joints = joint_state[:6]
        
        t_start = self.robot.initPeriod()
        self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)
        
        if self._use_gripper:
            # The last element is the gripper command
            self.gripper.move(joint_state[-1])
            
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        """Enable or disable manual teaching mode (Freedrive)."""
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
            log.info("Freedrive Enabled (Teach Mode)")
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()
            log.info("Freedrive Disabled (Control Mode)")

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Return a dictionary of robot observations."""
        joints = self.get_joint_state()
        
        # Try to get actual TCP pose from RTDE
        try:
            # RTDE provides getActualTCPPose() which returns [x, y, z, rx, ry, rz]
            # where rx, ry, rz are rotation vector (axis-angle representation)
            tcp_pose = self.r_inter.getActualTCPPose()
            if tcp_pose is not None and len(tcp_pose) >= 6:
                # Check if pose is valid (not all zeros)
                if not np.allclose(tcp_pose[:3], 0.0, atol=1e-6):
                    # Convert rotation vector to quaternion
                    # Rotation vector: [rx, ry, rz] where magnitude is rotation angle
                    rot_vec = tcp_pose[3:6]
                    angle = np.linalg.norm(rot_vec)
                    
                    if angle > 1e-6:  # Non-zero rotation
                        # Normalize rotation vector to get axis
                        axis = rot_vec / angle
                        # Convert axis-angle to quaternion: q = [cos(θ/2), sin(θ/2) * axis]
                        half_angle = angle / 2.0
                        qw = np.cos(half_angle)
                        qxyz = np.sin(half_angle) * axis
                        quat = np.array([qw, qxyz[0], qxyz[1], qxyz[2]])  # [w, x, y, z]
                    else:
                        # Zero rotation -> identity quaternion
                        quat = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
                    
                    # Store as [x, y, z, qx, qy, qz, qw] (position + quaternion)
                    # Note: Converting from [w, x, y, z] to [x, y, z, w] format
                    pos_quat = np.concatenate([tcp_pose[:3], quat[1:], [quat[0]]])
                else:
                    # TCP pose is all zeros, use zeros
                    pos_quat = np.zeros(7)
            else:
                # TCP pose is None or invalid, use zeros
                pos_quat = np.zeros(7)
        except Exception as e:
            # If getting TCP pose fails, log warning only once (to avoid spam)
            # Use a simple counter to limit logging frequency
            if not hasattr(self, '_tcp_pose_warn_count'):
                self._tcp_pose_warn_count = 0
            if self._tcp_pose_warn_count < 3:
                log.warn(f"Failed to get TCP pose from RTDE (will retry): {e}")
                self._tcp_pose_warn_count += 1
            pos_quat = np.zeros(7)
        
        # Get gripper position from joints (already normalized in get_joint_state)
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])
        
        # Try to get actual joint velocities from RTDE
        try:
            joint_velocities = self.r_inter.getActualQd()
            if joint_velocities is None or len(joint_velocities) == 0:
                joint_velocities = np.zeros_like(joints[:6])
            # Append gripper velocity (0 for now, as gripper doesn't provide velocity)
            if self._use_gripper:
                joint_velocities = np.append(joint_velocities, 0.0)
        except Exception as e:
            # If getting velocities fails, use zeros
            joint_velocities = np.zeros_like(joints)
        
        # Try to get TCP velocity from RTDE (linear and angular velocity)
        ee_velocity = np.zeros(3)  # Linear velocity [vx, vy, vz] in m/s
        ee_angular_velocity = np.zeros(3)  # Angular velocity [wx, wy, wz] in rad/s
        try:
            # RTDE provides getActualTCPSpeed() which returns [vx, vy, vz, wx, wy, wz]
            # where vx,vy,vz is linear velocity (m/s) and wx,wy,wz is angular velocity (rad/s)
            if hasattr(self.r_inter, 'getActualTCPSpeed'):
                tcp_speed = self.r_inter.getActualTCPSpeed()
                if tcp_speed is not None and len(tcp_speed) >= 6:
                    # Check if speed is valid (not all zeros or NaN)
                    if not (np.allclose(tcp_speed, 0.0, atol=1e-6) or np.any(np.isnan(tcp_speed))):
                        # Store linear velocity [vx, vy, vz] as ee_velocity (in m/s)
                        ee_velocity = tcp_speed[:3]
                        # Store angular velocity [wx, wy, wz] as ee_angular_velocity (in rad/s)
                        ee_angular_velocity = tcp_speed[3:6]
                    else:
                        ee_velocity = np.zeros(3)
                        ee_angular_velocity = np.zeros(3)
                else:
                    ee_velocity = np.zeros(3)
                    ee_angular_velocity = np.zeros(3)
            else:
                # RTDE version doesn't support getActualTCPSpeed(), use zeros
                ee_velocity = np.zeros(3)
                ee_angular_velocity = np.zeros(3)
        except Exception as e:
            # If getting TCP speed fails, use zeros
            # Note: Some RTDE versions may not support getActualTCPSpeed()
            if not hasattr(self, '_tcp_speed_warn_count'):
                self._tcp_speed_warn_count = 0
            if self._tcp_speed_warn_count < 1:
                log.warn(f"RTDE getActualTCPSpeed() not available or failed: {e}. Using zeros for EE velocity.")
                self._tcp_speed_warn_count += 1
            ee_velocity = np.zeros(3)
            ee_angular_velocity = np.zeros(3)
        
        return {
            "joint_positions": joints,
            "joint_velocities": joint_velocities,
            "ee_pos_quat": pos_quat,
            "ee_velocity": ee_velocity,  # TCP linear velocity [vx, vy, vz] in m/s
            "ee_angular_velocity": ee_angular_velocity,  # TCP angular velocity [wx, wy, wz] in rad/s
            "gripper_position": gripper_pos,
        }