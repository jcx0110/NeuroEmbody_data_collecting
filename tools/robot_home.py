#!/usr/bin/env python3
"""
Robot Home Position Script
Detects robot connection and moves robot to home position.
"""

import sys
import time
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from data_collecting.utils.logger import Logger as log
    from data_collecting.utils.config_loader import ConfigLoader
    from data_collecting.robot.ur import URRobot
except ImportError as e:
    print(f"Error: Failed to import required modules: {e}")
    print("Please ensure you are running from the project root directory.")
    sys.exit(1)

# Default home position (in radians)
DEFAULT_HOME_JOINTS = [-1.57, -1.57, -1.57, -1.57, 1.57, 3.14]

def check_robot_connection(robot_ip: str) -> bool:
    """
    Check if robot is reachable at the given IP address.
    
    Args:
        robot_ip: IP address of the robot
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        import rtde_receive
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        # Try to get current joint positions
        joints = rtde_r.getActualQ()
        rtde_r.disconnect()
        return True
    except Exception as e:
        log.error(f"Failed to connect to robot at {robot_ip}: {e}")
        return False

def move_to_home(robot_ip: str, home_joints: list = None, speed: float = 0.5, acceleration: float = 0.5) -> bool:
    """
    Move robot to home position.
    
    Args:
        robot_ip: IP address of the robot
        home_joints: Target joint positions (default: DEFAULT_HOME_JOINTS)
        speed: Movement speed (rad/s)
        acceleration: Movement acceleration (rad/s^2)
        
    Returns:
        True if successful, False otherwise
    """
    if home_joints is None:
        home_joints = DEFAULT_HOME_JOINTS
    
    try:
        import rtde_control
        import rtde_receive
        
        log.info(f"Connecting to robot at {robot_ip}...")
        rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        
        # Get current joint positions
        current_joints = rtde_r.getActualQ()
        log.info(f"Current joint positions: {[f'{j:.3f}' for j in current_joints]}")
        
        # Calculate distance to home
        import numpy as np
        joint_diff = np.array(home_joints) - np.array(current_joints)
        max_diff = np.max(np.abs(joint_diff))
        
        if max_diff < 0.01:
            log.info("Robot is already at home position.")
            rtde_c.disconnect()
            rtde_r.disconnect()
            return True
        
        log.info(f"Moving to home position...")
        log.info(f"Target joint positions: {[f'{j:.3f}' for j in home_joints]}")
        
        # Move to home position
        rtde_c.moveJ(home_joints, speed, acceleration)
        
        # Wait for movement to complete
        time.sleep(0.5)
        
        # Verify final position
        final_joints = rtde_r.getActualQ()
        final_diff = np.array(home_joints) - np.array(final_joints)
        final_max_diff = np.max(np.abs(final_diff))
        
        if final_max_diff < 0.05:
            log.success(f"Robot moved to home position successfully.")
            log.info(f"Final joint positions: {[f'{j:.3f}' for j in final_joints]}")
        else:
            log.warn(f"Robot may not have reached exact home position (max diff: {final_max_diff:.3f} rad)")
        
        rtde_c.disconnect()
        rtde_r.disconnect()
        return True
        
    except Exception as e:
        log.error(f"Failed to move robot to home position: {e}")
        return False

def main():
    """Main function to check connection and move robot to home."""
    
    # Load configuration
    cfg_loader = ConfigLoader()
    robot_cfg = cfg_loader.get_robot()
    robot_ip = robot_cfg.get("ip", "192.168.1.2")
    robot_type = robot_cfg.get("type", "ur")
    
    log.info("=" * 60)
    log.info("Robot Home Position Script")
    log.info("=" * 60)
    log.info(f"Robot Type: {robot_type}")
    log.info(f"Robot IP: {robot_ip}")
    log.info("")
    
    # Check robot connection
    log.info("Step 1: Checking robot connection...")
    if not check_robot_connection(robot_ip):
        log.error("Robot connection check failed. Please verify:")
        log.error("  1. Robot is powered on")
        log.error("  2. Network connection is active")
        log.error("  3. Robot IP address is correct")
        log.error("  4. Firewall is not blocking the connection")
        sys.exit(1)
    
    log.success("Robot connection successful.")
    log.info("")
    
    # Move to home position
    log.info("Step 2: Moving robot to home position...")
    if not move_to_home(robot_ip):
        log.error("Failed to move robot to home position.")
        sys.exit(1)
    
    log.info("")
    log.success("Script completed successfully.")
    log.info("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.warn("\nScript interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        sys.exit(1)

