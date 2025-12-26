#!/usr/bin/env python3
"""
Simple RealSense Camera Test Script
Tests camera initialization and frame reading.
"""

import sys
import time
from pathlib import Path

# Add project root to path
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    import pyrealsense2 as rs
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install pyrealsense2 opencv-python numpy")
    sys.exit(1)

def test_camera(device_id=None, width=640, height=480, fps=30, num_frames=10):
    """
    Test RealSense camera initialization and frame reading.
    
    Args:
        device_id: Camera device ID (serial number) or None for auto-select
        width: Image width
        height: Image height
        fps: Frames per second
        num_frames: Number of test frames to capture
    """
    print("=" * 60)
    print("RealSense Camera Test")
    print("=" * 60)
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure device if specified
    if device_id and device_id != "default":
        print(f"Configuring device: {device_id}")
        config.enable_device(str(device_id))
    else:
        print("Auto-selecting first available device")
    
    # Configure streams
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    
    # Alignment
    align = rs.align(rs.stream.color)
    
    try:
        # Start pipeline
        print("\nStep 1: Starting pipeline...")
        profile = pipeline.start(config)
        print("Pipeline started successfully")
        
        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(f"Device: {device_name}")
        print(f"Serial: {serial_number}")
        
        # Wait for camera to stabilize (no frame reading during this time)
        print("\nStep 2: Waiting for camera to stabilize...")
        print("Waiting 3 seconds for camera initialization...")
        time.sleep(3.0)
        
        # Optional: Try to get one frame to verify camera is working
        print("Verifying camera is ready (testing first frame)...")
        try:
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            print("Camera is ready (first frame received successfully)")
        except RuntimeError as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower() or "didn't arrive" in error_msg.lower():
                print("Warning: First frame timeout, but continuing test...")
                print("This may indicate camera needs more time or has connection issues")
            else:
                print(f"Error: {error_msg}")
                raise
        
        # Test frame reading
        print(f"\nStep 3: Testing frame capture ({num_frames} frames)...")
        success_count = 0
        fail_count = 0
        
        for i in range(num_frames):
            try:
                # Wait for frames with timeout
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                
                # Align depth to color
                aligned_frames = align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    print(f"  Frame {i+1}: OK - Color: {color_image.shape}, Depth: {depth_image.shape}")
                    success_count += 1
                else:
                    print(f"  Frame {i+1}: Failed - Missing frames")
                    fail_count += 1
                    
            except RuntimeError as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "didn't arrive" in error_msg.lower():
                    print(f"  Frame {i+1}: Timeout - {error_msg}")
                else:
                    print(f"  Frame {i+1}: Error - {error_msg}")
                fail_count += 1
            except Exception as e:
                print(f"  Frame {i+1}: Unexpected error - {e}")
                fail_count += 1
            
            # Small delay between frames (not needed, but helps with timing)
            # time.sleep(0.1)  # Commented out to test at full speed
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Successful frames: {success_count}/{num_frames}")
        print(f"Failed frames: {fail_count}/{num_frames}")
        print(f"Success rate: {success_count/num_frames*100:.1f}%")
        
        if success_count == num_frames:
            print("\nResult: PASS - All frames captured successfully")
            return True
        elif success_count >= num_frames * 0.8:
            print("\nResult: WARNING - Some frames failed, but camera is mostly working")
            return True
        else:
            print("\nResult: FAIL - Too many frame failures")
            return False
            
    except Exception as e:
        print(f"\nError during test: {e}")
        return False
        
    finally:
        # Stop pipeline
        print("\nStopping pipeline...")
        try:
            pipeline.stop()
            print("Pipeline stopped")
        except:
            pass
        
        print("=" * 60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test RealSense camera")
    parser.add_argument("--device_id", type=str, default=None,
                        help="Camera device ID (serial number). Use 'default' or omit for auto-select")
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--num_frames", type=int, default=10, help="Number of test frames")
    
    args = parser.parse_args()
    
    # Handle "default" string
    device_id = None if args.device_id == "default" else args.device_id
    
    success = test_camera(
        device_id=device_id,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

