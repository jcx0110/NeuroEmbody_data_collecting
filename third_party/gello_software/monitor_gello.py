import time
import yaml
import numpy as np
from gello.agents.gello_agent import GelloAgent

# Point to the auto-generated config file
CONFIG_PATH = "configs/yam_auto_generated.yaml"

def main():
    print(f"Loading configuration: {CONFIG_PATH} ...")
    
    # 1. Read YAML configuration
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Extract key parameters
    agent_config = config["agent"]
    port = agent_config["port"]
    # Note: YAML may read as list, ensure it's tuple or list
    offsets = agent_config["joint_offsets"] 
    signs = agent_config["joint_signs"]

    print(f"Connecting to port: {port}")
    print(f"Loading zero positions: {offsets}")

    # 2. Initialize Agent (only connect to handle, not arm)
    # We directly instantiate GelloAgent, which only reads Dynamixel
    agent = GelloAgent(port=port, start_joints=offsets, joint_signs=signs)

    print("\nGELLO handle initialized successfully!")
    print("--------------------------------------------------")
    print("Reading data (Press Ctrl+C to exit)...")
    print("--------------------------------------------------")

    try:
        while True:
            # Read current joint angles
            joints = agent.act(None) # act method returns current joint state
            
            # Format print (keep 3 decimal places)
            formatted_joints = [f"{j:.3f}" for j in joints]
            print(f"\rJoint angles: {formatted_joints}", end="")
            
            time.sleep(0.1) # 10Hz refresh rate, no need to be too fast

    except KeyboardInterrupt:
        print("\n\nStopped.")
        # The agent here doesn't have an explicit close method, Python will automatically release the serial port on exit
        
if __name__ == "__main__":
    main()
