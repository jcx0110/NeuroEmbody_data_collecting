import sys
import time
import threading
from pathlib import Path
from pymodbus.client import ModbusTcpClient

# Path handling to ensure logger can be imported
try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log

class MyGripper:
    def __init__(self, ip_address='192.168.1.11', port=502):
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.lock = threading.Lock()
        
        # Record the last command to prevent duplicate/spamming commands
        self.last_target_pos = -1 
        
        # Current normalized position (0.0 - 1.0) for get_current_position
        self.current_normalized_pos = 0.0
        
        self.connect()
        self.activate_gripper()

    def connect(self):
        """Establish Modbus connection to the gripper."""
        try:
            if self.client:
                self.client.close()
            
            self.client = ModbusTcpClient(self.ip_address, port=self.port)
            
            if self.client.connect():
                log.success(f"[Gripper] Connected to {self.ip_address}")
            else:
                log.error(f"[Gripper] Connection failed to {self.ip_address}")
                
        except Exception as e:
            log.error(f"[Gripper] Connection Error: {e}")

    def activate_gripper(self):
        """Send initialization/activation command sequence."""
        try:
            # Activation registers: 0x0100 (Activate), 0x0000, 0x6464 ...
            # Writing to register 0
            self.client.write_registers(0, [0x0100, 0x0000, 0x6464, 0, 0, 0, 0, 0])
            time.sleep(0.1) # Small delay for initialization
            log.info("[Gripper] Activation command sent.")
        except Exception as e:
            log.warn(f"[Gripper] Activation failed: {e}")

    def move(self, value: float):
        """
        Main function called by URRobot (Gello interface).
        :param value: 0.0 (Open) to 1.0 (Closed)
        """
        # 1. Clamp value to 0.0 - 1.0 range
        val_clamped = max(0.0, min(1.0, value))
        self.current_normalized_pos = val_clamped

        # 2. Convert to 0-255 (Gripper internal range)
        target_int = int(val_clamped * 255)

        # 3. Filter noise: Only send if change is significant (> 2 units)
        # This prevents flooding the Modbus network with tiny fluctuations
        if abs(target_int - self.last_target_pos) > 2:
            self._send_cmd(target_int)
            self.last_target_pos = target_int

    def _send_cmd(self, position_int):
        """Send Modbus command (Non-blocking as much as possible)."""
        try:
            # Command structure: [0x0900 (Go to), position, speed/force, ...]
            cmd = [0x0900, position_int, 0x6464, 0, 0, 0, 0, 1]
            self.client.write_registers(0, cmd)
        except Exception:
            # Silent retry mechanism to avoid crashing the main control loop
            try:
                self.connect()
            except:
                pass

    def get_current_position(self) -> float:
        """
        Get current position (0-255).
        NOTE: To avoid blocking the UR 500Hz loop, we return the cached 
        last command value instead of querying over the network.
        """
        return self.current_normalized_pos * 255.0

# ================= Self-Test Code =================
if __name__ == "__main__":
    print("\n--- Gripper Self-Test ---")
    try:
        # Note: Change IP to your actual gripper IP for testing
        gripper = MyGripper(ip_address='192.168.1.11')
        
        print("Closing (1.0)...")
        gripper.move(1.0)
        time.sleep(2)
        
        print("Opening (0.0)...")
        gripper.move(0.0)
        time.sleep(2)
        
        print("Test Complete.")
    except Exception as e:
        print(f"Test Failed: {e}")