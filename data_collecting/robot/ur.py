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
        
        # Placeholder for EE pose if needed later
        pos_quat = np.zeros(7) 
        
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])
        
        return {
            "joint_positions": joints,
            "joint_velocities": joints, # Warning: This seems to copy positions; consider getting actual velocities if needed
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }