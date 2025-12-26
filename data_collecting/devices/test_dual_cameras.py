#!/usr/bin/env python3
"""
Test Dual RealSense Cameras
Tests each camera individually and then together to identify issues.
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
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install pyrealsense2 numpy")
    sys.exit(1)

def test_single_camera(device_id, camera_name, num_frames=5):
    """Test a single camera"""
    print(f"\n{'='*60}")
    print(f"Testing {camera_name} Camera (Device ID: {device_id})")
    print(f"{'='*60}")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(str(device_id))
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    align = rs.align(rs.stream.color)
    
    try:
        print("Starting pipeline...")
        profile = pipeline.start(config)
        device = profile.get_device()
        print(f"Device: {device.get_info(rs.camera_info.name)}")
        print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
        
        print("Waiting 3 seconds for camera to stabilize...")
        time.sleep(3.0)
        
        print(f"Testing frame capture ({num_frames} frames)...")
        success = 0
        for i in range(num_frames):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=10000)
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    print(f"  Frame {i+1}: OK")
                    success += 1
                else:
                    print(f"  Frame {i+1}: Failed - Missing frames")
            except RuntimeError as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower() or "didn't arrive" in error_msg.lower():
                    print(f"  Frame {i+1}: Timeout")
                else:
                    print(f"  Frame {i+1}: Error - {error_msg}")
        
        print(f"\nResult: {success}/{num_frames} frames successful")
        pipeline.stop()
        return success == num_frames
        
    except Exception as e:
        print(f"Error: {e}")
        try:
            pipeline.stop()
        except:
            pass
        return False

def test_both_cameras(front_id, side_id, num_frames=5):
    """Test both cameras together"""
    print(f"\n{'='*60}")
    print("Testing Both Cameras Together")
    print(f"{'='*60}")
    
    # Initialize first camera
    print("\nInitializing front camera...")
    front_pipeline = rs.pipeline()
    front_config = rs.config()
    front_config.enable_device(str(front_id))
    front_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    front_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    front_align = rs.align(rs.stream.color)
    
    try:
        front_profile = front_pipeline.start(front_config)
        print("Front camera started")
        print("Waiting 3 seconds for front camera to stabilize...")
        time.sleep(3.0)
    except Exception as e:
        print(f"Failed to start front camera: {e}")
        return False
    
    # Initialize second camera after delay
    print("\nInitializing side camera...")
    side_pipeline = rs.pipeline()
    side_config = rs.config()
    side_config.enable_device(str(side_id))
    side_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    side_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    side_align = rs.align(rs.stream.color)
    
    try:
        side_profile = side_pipeline.start(side_config)
        print("Side camera started")
        print("Waiting 2 seconds for side camera to stabilize...")
        time.sleep(2.0)
    except Exception as e:
        print(f"Failed to start side camera: {e}")
        front_pipeline.stop()
        return False
    
    # Test reading from both
    print(f"\nTesting simultaneous frame capture ({num_frames} frames)...")
    front_success = 0
    side_success = 0
    
    for i in range(num_frames):
        # Read from front camera
        try:
            frames = front_pipeline.wait_for_frames(timeout_ms=10000)
            aligned_frames = front_align.process(frames)
            if aligned_frames.get_color_frame() and aligned_frames.get_depth_frame():
                front_success += 1
                print(f"  Frame {i+1}: Front OK", end="")
            else:
                print(f"  Frame {i+1}: Front Failed", end="")
        except RuntimeError:
            print(f"  Frame {i+1}: Front Timeout", end="")
        
        # Read from side camera
        try:
            frames = side_pipeline.wait_for_frames(timeout_ms=10000)
            aligned_frames = side_align.process(frames)
            if aligned_frames.get_color_frame() and aligned_frames.get_depth_frame():
                side_success += 1
                print(" | Side OK")
            else:
                print(" | Side Failed")
        except RuntimeError:
            print(" | Side Timeout")
        
        time.sleep(0.1)
    
    print(f"\nResults:")
    print(f"  Front camera: {front_success}/{num_frames} frames")
    print(f"  Side camera: {side_success}/{num_frames} frames")
    
    front_pipeline.stop()
    side_pipeline.stop()
    
    return front_success == num_frames and side_success == num_frames

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dual RealSense cameras")
    parser.add_argument("--front_id", type=str, required=True, help="Front camera device ID")
    parser.add_argument("--side_id", type=str, required=True, help="Side camera device ID")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of test frames")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Dual Camera Test")
    print("="*60)
    
    # Test each camera individually
    print("\nPHASE 1: Testing cameras individually")
    front_ok = test_single_camera(args.front_id, "Front", args.num_frames)
    time.sleep(2.0)  # Wait between tests
    
    side_ok = test_single_camera(args.side_id, "Side", args.num_frames)
    time.sleep(2.0)  # Wait between tests
    
    # Test both together
    print("\nPHASE 2: Testing cameras together")
    both_ok = test_both_cameras(args.front_id, args.side_id, args.num_frames)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Front camera (individual): {'PASS' if front_ok else 'FAIL'}")
    print(f"Side camera (individual): {'PASS' if side_ok else 'FAIL'}")
    print(f"Both cameras (together): {'PASS' if both_ok else 'FAIL'}")
    
    if front_ok and side_ok and not both_ok:
        print("\nDiagnosis: Both cameras work individually but fail together.")
        print("This suggests USB bandwidth or resource competition issues.")
        print("Recommendations:")
        print("  1. Use USB 3.0 ports (not USB 2.0)")
        print("  2. Connect cameras to different USB controllers if possible")
        print("  3. Reduce resolution or frame rate")
        print("  4. Add delays between camera initializations")
    elif not front_ok:
        print("\nDiagnosis: Front camera has issues even when used alone.")
    elif not side_ok:
        print("\nDiagnosis: Side camera has issues even when used alone.")
    elif both_ok:
        print("\nDiagnosis: Both cameras work correctly individually and together.")
    
    print("="*60)

if __name__ == "__main__":
    main()

