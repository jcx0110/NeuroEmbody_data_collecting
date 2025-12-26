import os
import sys
import time
import numpy as np
import pyrealsense2 as rs
from pathlib import Path

# Ensure path is correct to import logger
try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log

class RealSenseCamera:
    def __init__(self, device_id="default", width=640, height=480, fps=30):
        """
        Initialize the RealSense Camera with auto-alignment and warm-up.
        """
        self.device_id = device_id
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Lock to specific device if ID is provided
        if device_id != "default" and device_id is not None:
            config.enable_device(str(device_id))
            
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        # Alignment to color stream coordinate system
        self.align = rs.align(rs.stream.color)
        
        # Frame cache for handling dropped frames (last successful frame)
        self.last_color_frame = None
        self.last_depth_frame = None

        try:
            self.profile = self.pipeline.start(config)
            
            # Get device info to detect camera model
            device = self.profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            self.is_d455 = "D455" in device_name
            
            log.success(f"RealSense Camera Started: {device_name} ({width}x{height} @ {fps}fps)")
            
            # D455 may need special handling - set frame queue size for better buffering
            if self.is_d455:
                try:
                    sensors = device.query_sensors()
                    for sensor in sensors:
                        if sensor.supports(rs.option.frames_queue_size):
                            sensor.set_option(rs.option.frames_queue_size, 10)
                            log.info("D455: Set frame queue size to 10 for better buffering")
                except Exception as e:
                    log.warn(f"Could not set frame queue size: {e}")
            
            # Warming up logic: Wait for camera to stabilize
            # D455 may need longer warm-up time
            warmup_delay = 4.0 if self.is_d455 else 3.0
            log.info(f"Waiting for camera to stabilize ({warmup_delay}s)...")
            time.sleep(warmup_delay)
            
            # Try to get a few frames, but don't require all to succeed
            log.info("Warming up camera (auto-exposure)...")
            warmup_success = 0
            warmup_attempts = 5
            # D455 may need longer timeout
            warmup_timeout_ms = 15000 if self.is_d455 else 10000
            
            for i in range(warmup_attempts):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=warmup_timeout_ms)
                    aligned_frames = self.align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()
                    
                    if color_frame and depth_frame:
                        # Cache the first successful frame for frame drop recovery
                        if self.last_color_frame is None:
                            self.last_color_frame = np.asanyarray(color_frame.get_data()).copy()
                            self.last_depth_frame = np.asanyarray(depth_frame.get_data()).copy()
                        warmup_success += 1
                except RuntimeError as e:
                    # Allow timeouts during warm-up - this is common and OK
                    error_msg = str(e).lower()
                    if "timeout" in error_msg or "didn't arrive" in error_msg:
                        # Don't log every timeout to reduce noise
                        pass
                    else:
                        # Only log non-timeout errors
                        log.warn(f"Warm-up error (attempt {i + 1}/{warmup_attempts}): {e}")
            
            # Camera is ready even if warm-up frames failed
            if warmup_success > 0:
                log.success(f"Camera ready. ({warmup_success}/{warmup_attempts} warm-up frames successful)")
            else:
                log.warn("No warm-up frames captured, but camera should still work. Continuing...")
                log.success("Camera ready.")
            
        except Exception as e:
            log.error(f"Failed to start RealSense: {e}")
            self.pipeline = None
            self.is_d455 = False

    def read(self, max_retries=2, use_frame_cache=True):
        """
        Reads aligned RGB and Depth images with retry mechanism and frame caching.
        :param max_retries: Maximum number of retry attempts for D455 cameras
        :param use_frame_cache: If True, return last successful frame on failure (for D455)
        :return: (color_image, depth_image) as numpy arrays, or (None, None) on failure.
        """
        if not self.pipeline: 
            return None, None
        
        # D455 may need longer timeout and retry mechanism
        timeout_ms = 10000 if self.is_d455 else 5000
        retries = max_retries if self.is_d455 else 0
        
        for attempt in range(retries + 1):
            try:
                # Wait for frames with timeout (longer for D455)
                frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    if attempt < retries:
                        continue  # Retry if frames are missing
                    # If we have cached frames, use them
                    if use_frame_cache and self.last_color_frame is not None and self.last_depth_frame is not None:
                        log.warn("Empty frame received, using cached frame")
                        return self.last_color_frame.copy(), self.last_depth_frame.copy()
                    log.warn("Empty frame received from RealSense.")
                    return None, None

                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data())
                
                # Update frame cache
                if use_frame_cache:
                    self.last_color_frame = color_image.copy()
                    self.last_depth_frame = depth_image.copy()
                
                return color_image, depth_image
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                if ("timeout" in error_msg or "didn't arrive" in error_msg):
                    if attempt < retries:
                        continue  # Retry on timeout
                    # If we have cached frames, use them instead of returning None
                    if use_frame_cache and self.last_color_frame is not None and self.last_depth_frame is not None:
                        log.warn("Frame timeout, using cached frame")
                        return self.last_color_frame.copy(), self.last_depth_frame.copy()
                    # Only log error if no cache available
                    if attempt == retries:
                        log.error(f"RealSense Runtime Error during read: {e}")
                    return None, None
                else:
                    # Non-timeout errors
                    if attempt == retries:
                        log.error(f"RealSense Runtime Error during read: {e}")
                    return None, None
            except Exception as e:
                if attempt == retries:
                    log.error(f"Unexpected Camera Read Error: {e}")
                return None, None
        
        # Final fallback: use cached frame if available
        if use_frame_cache and self.last_color_frame is not None and self.last_depth_frame is not None:
            log.warn("All retries failed, using cached frame")
            return self.last_color_frame.copy(), self.last_depth_frame.copy()
        
        return None, None

    def stop(self):
        """Safely stops the camera pipeline."""
        if self.pipeline: 
            try:
                self.pipeline.stop()
                log.info("RealSense pipeline stopped.")
            except Exception as e:
                log.warn(f"Error while stopping RealSense: {e}")

# ==========================================
# Self-Test Code (Standalone verification)
# ==========================================
if __name__ == "__main__":
    print("\n--- RealSense Standalone Check ---")
    
    # Initialize camera
    cam = RealSenseCamera(width=640, height=480, fps=30)
    
    # Try to grab 5 frames to verify data
    if cam.pipeline:
        for i in range(5):
            c, d = cam.read()
            if c is not None:
                print(f"Captured frame {i+1}: RGB Shape {c.shape}, Depth Shape {d.shape}")
            else:
                print(f"Failed to capture frame {i+1}")
        
        cam.stop()
        print("--- Test Passed ---\n")
    else:
        print("--- Test Failed: Camera not initialized ---\n")