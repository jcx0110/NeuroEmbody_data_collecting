from typing import Dict
import numpy as np
import cv2
from gello.robots.robot import Robot

class URRobot(Robot):
    def __init__(self, robot_ip: str = "192.168.1.10", no_gripper: bool = False):
        import rtde_control
        import rtde_receive

        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(f"UR Connection Error: {e}")

        self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        self._use_gripper = not no_gripper
        if self._use_gripper:
            # Modification point 1: Import your new file
            from gello.robots.my_gripper import MyGripper
            
            # Modification point 2: Fill in your gripper's actual IP (192.168.1.11)
            print("Connecting to Custom Gripper...")
            self.gripper = MyGripper(ip_address='192.168.1.11') 

        self._free_drive = False
        self.robot.endFreedriveMode()

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            # Simplified read, directly return 0 to avoid read lag
            pos = np.append(robot_joints, 0.0) 
        else:
            pos = robot_joints
        return pos

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        velocity = 0.5
        acceleration = 0.5
        dt = 1.0 / 500
        lookahead_time = 0.2
        gain = 100

        robot_joints = joint_state[:6]
        t_start = self.robot.initPeriod()
        self.robot.servoJ(robot_joints, velocity, acceleration, dt, lookahead_time, gain)
        
        if self._use_gripper:
            # Modification point 3: Use simple move(val) call
            # joint_state[-1] is a float between 0.0~1.0
            self.gripper.move(joint_state[-1])
            
        self.robot.waitPeriod(t_start)

    def freedrive_enabled(self) -> bool:
        return self._free_drive

    def set_freedrive_mode(self, enable: bool) -> None:
        if enable and not self._free_drive:
            self._free_drive = True
            self.robot.freedriveMode()
        elif not enable and self._free_drive:
            self._free_drive = False
            self.robot.endFreedriveMode()

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        
        # Get actual TCP pose from robot (position + quaternion)
        # getActualTCPPose returns [x, y, z, rx, ry, rz] where rx,ry,rz is rotation vector
        # We need to convert rotation vector to quaternion
        try:
            tcp_pose = self.r_inter.getActualTCPPose()
            # tcp_pose is [x, y, z, rx, ry, rz] - rotation vector format
            pos = np.array(tcp_pose[:3])  # Position [x, y, z]
            
            # Convert rotation vector to quaternion using cv2
            rot_vec = np.array(tcp_pose[3:6], dtype=np.float64)
            rot_mat, _ = cv2.Rodrigues(rot_vec)  # Convert rotation vector to rotation matrix
            
            # Convert rotation matrix to quaternion
            # Using standard method: qw = sqrt(1 + trace(R)) / 2
            trace = np.trace(rot_mat)
            if trace > 0:
                s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
                qw = 0.25 * s
                qx = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                qy = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                qz = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            else:
                # Handle case when trace <= 0
                if rot_mat[0, 0] > rot_mat[1, 1] and rot_mat[0, 0] > rot_mat[2, 2]:
                    s = np.sqrt(1.0 + rot_mat[0, 0] - rot_mat[1, 1] - rot_mat[2, 2]) * 2
                    qx = 0.25 * s
                    qy = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                    qz = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                    qw = (rot_mat[2, 1] - rot_mat[1, 2]) / s
                elif rot_mat[1, 1] > rot_mat[2, 2]:
                    s = np.sqrt(1.0 + rot_mat[1, 1] - rot_mat[0, 0] - rot_mat[2, 2]) * 2
                    qx = (rot_mat[0, 1] + rot_mat[1, 0]) / s
                    qy = 0.25 * s
                    qz = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                    qw = (rot_mat[0, 2] - rot_mat[2, 0]) / s
                else:
                    s = np.sqrt(1.0 + rot_mat[2, 2] - rot_mat[0, 0] - rot_mat[1, 1]) * 2
                    qx = (rot_mat[0, 2] + rot_mat[2, 0]) / s
                    qy = (rot_mat[1, 2] + rot_mat[2, 1]) / s
                    qz = 0.25 * s
                    qw = (rot_mat[1, 0] - rot_mat[0, 1]) / s
            
            quat = np.array([qx, qy, qz, qw])  # [qx, qy, qz, qw] format
            
            # Combine position and quaternion: [x, y, z, qx, qy, qz, qw]
            pos_quat = np.concatenate([pos, quat])
        except Exception as e:
            # Fallback to zeros if getting TCP pose fails
            print(f"Warning: Failed to get TCP pose: {e}")
            pos_quat = np.zeros(7)
        
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])
        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }