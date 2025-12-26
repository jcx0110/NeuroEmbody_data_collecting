#!/usr/bin/env python3
"""
Tool script: List all available RealSense camera devices
Usage: python tools/list_realsense_cameras.py
"""

import sys
from pathlib import Path

# Add project root directory to path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import pyrealsense2 as rs
except ImportError:
    print("Error: pyrealsense2 is not installed")
    print("Please run: pip install pyrealsense2")
    sys.exit(1)

def list_realsense_devices():
    """List all connected RealSense devices"""
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense camera devices detected")
        print("\nPlease check:")
        print("  1. Whether the camera is properly connected to USB port")
        print("  2. Whether the USB cable supports data transfer (not charging-only)")
        print("  3. Whether RealSense SDK is installed")
        return
    
    print(f"\nDetected {len(devices)} RealSense device(s):\n")
    print("=" * 80)
    
    for idx, dev in enumerate(devices):
        try:
            serial_number = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            firmware_version = dev.get_info(rs.camera_info.firmware_version)
            
            print(f"\nDevice #{idx + 1}:")
            print(f"  Name: {name}")
            print(f"  Serial Number: {serial_number}")
            print(f"  Firmware Version: {firmware_version}")
            print(f"\n  Tip: To use this device, set device_id = '{serial_number}'")
            print("-" * 80)
            
        except Exception as e:
            print(f"\nDevice #{idx + 1}: Failed to read information - {e}")
            print("-" * 80)
    
    print("\nUsage:")
    print("  Method 1: Command line arguments")
    print("    python data_collecting/core/run_data_collecting.py \\")
    print("      --front_camera_id <serial_number1> \\")
    print("      --side_camera_id <serial_number2>")
    print()
    print("  Method 2: Set directly in code")
    print("    front_cam = RealSenseCamera(device_id='<serial_number1>', ...)")
    print("    side_cam = RealSenseCamera(device_id='<serial_number2>', ...)")
    print()
    print("  Method 3: Use 'default' (auto-select first available device)")
    print("    front_cam = RealSenseCamera(device_id='default', ...)")
    print()

if __name__ == "__main__":
    list_realsense_devices()
