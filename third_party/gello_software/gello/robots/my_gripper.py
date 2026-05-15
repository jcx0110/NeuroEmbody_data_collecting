from pymodbus.client import ModbusTcpClient
import threading
import time

class MyGripper:
    def __init__(self, ip_address='192.168.1.11', port=502):
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.lock = threading.Lock()
        
        # Record last command position to prevent duplicate sends
        self.last_target_pos = -1 
        # Record current normalized position (0.0 - 1.0) for get_current_position
        self.current_normalized_pos = 0.0
        
        self.connect()
        self.activate_gripper()

    def connect(self):
        try:
            if self.client:
                self.client.close()
            self.client = ModbusTcpClient(self.ip_address, port=self.port)
            if self.client.connect():
                print(f"[Gripper] Connected to {self.ip_address}")
            else:
                print(f"[Gripper] Connection failed to {self.ip_address}")
        except Exception as e:
            print(f"[Gripper] Error: {e}")

    def activate_gripper(self):
        """Send activation command"""
        try:
            # Your activation command: 0x0100 (Activate), 0x0000, 0x6464 ...
            # Write to register 0
            self.client.write_registers(0, [0x0100, 0x0000, 0x6464, 0, 0, 0, 0, 0])
            time.sleep(0.1) # Wait a bit for initialization
            print("[Gripper] Activated.")
        except Exception as e:
            print(f"[Gripper] Activation failed: {e}")

    def move(self, value: float):
        """
        Main function called by GELLO
        :param value: 0.0 (Open) ~ 1.0 (Closed)
        """
        # 1. Clamp range to 0.0 - 1.0
        val_clamped = max(0.0, min(1.0, value))
        self.current_normalized_pos = val_clamped

        # 2. Convert to 0-255
        target_int = int(val_clamped * 255)

        # 3. Only send if change is significant (filter noise, prevent Modbus congestion)
        if abs(target_int - self.last_target_pos) > 2:
            self._send_cmd(target_int)
            self.last_target_pos = target_int

    def _send_cmd(self, position_int):
        """Send Modbus command (non-blocking)"""
        try:
            # Your movement command: [0x0900, position, 0x6464, ...]
            # Note: position must be int
            cmd = [0x0900, position_int, 0x6464, 0, 0, 0, 0, 1]
            self.client.write_registers(0, cmd)
        except Exception:
            # If send fails, try silent reconnect, don't crash with error
            try:
                self.connect()
            except:
                pass

    def get_current_position(self) -> float:
        """
        Get current position (0-255)
        Note: To avoid blocking UR's 500Hz loop, we don't read Modbus over network here,
        but directly return the last command position we sent.
        """
        # Return value between 0-255
        return self.current_normalized_pos * 255.0
