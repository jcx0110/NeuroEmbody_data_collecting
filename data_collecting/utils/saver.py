import os
import sys
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

# Path handling: Ensure independent execution works by adding project root to sys.path
try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log

class DataSaver:
    def __init__(self, save_dir, task_name, description=""):
        self.task_name = task_name
        self.description = description
        
        # Structure optimization: Automatically create a subdirectory named after task_name under save_dir
        self.output_dir = os.path.join(save_dir, task_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            log.info(f"Initialized storage directory: {self.output_dir}")
        else:
            log.info(f"Storage directory exists: {self.output_dir}")

    def save_episode(self, buffer, episode_idx, task_descriptions=None, task_switch_frames=[], expected_stages=4):
        """
        Save a single episode to an HDF5 file.
        :param buffer: Dictionary containing collected data
        :param episode_idx: Index of the current episode
        :param task_descriptions: List of strings describing task stages (e.g., ["pick up", "move to"])
        """

        # Safety check: Ensure the buffer is not empty before proceeding
        # Check for new dual-camera format first, fallback to old format
        colors_front = buffer.get('colors_front')
        colors_side = buffer.get('colors_side')
        colors_concate = buffer.get('colors_concate')
        
        # Fallback to old format for backward compatibility
        if colors_front is None or (isinstance(colors_front, list) and len(colors_front) == 0):
            colors = buffer.get('colors')  # Old format
            if colors is None or (isinstance(colors, list) and len(colors) == 0):
                log.error("Buffer is empty, skipping save.")
                return False
            # Use old format
            use_dual_camera = False
        else:
            use_dual_camera = True
        
        # Stage check: Check if the number of recorded switches matches the expected stages in config
        actual_stages = len(task_switch_frames)
        if actual_stages != expected_stages:
            log.error("VALIDATION FAILED: Stage count mismatch!")
            log.error(f"  -> Config expects: {expected_stages} stages")
            log.error(f"  -> Actually recorded: {actual_stages} switches")
            log.error(f"  -> DATA DISCARDED for Episode {episode_idx}")
            return False

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"episode_{episode_idx:03d}.h5"
        path = os.path.join(self.output_dir, filename)

        # Prepare Data: Process raw data in buffer to match the target storage format
        # Assuming buffer['ee_poses'] is N x 7 (x,y,z, qx,qy,qz,qw)
        if use_dual_camera:
            num_frames = len(colors_front)
        else:
            num_frames = len(colors)
            
        ee_poses = np.array(buffer.get('ee_poses', []))
        if len(ee_poses) > 0 and ee_poses.shape[1] == 7:
            ee_position = ee_poses[:, :3]
            ee_pose_quat = ee_poses[:, 3:]
        else:
            # Fallback if pose data is missing or incorrect
            ee_position = np.zeros((num_frames, 3))
            ee_pose_quat = np.zeros((num_frames, 4))

        try:
            with h5py.File(path, "w") as f:

                # 1. Image Data 
                if use_dual_camera:
                    # New dual-camera format
                    f.create_dataset("color_frames_front", data=np.array(colors_front, dtype=np.uint8), compression="gzip")
                    f.create_dataset("color_frames_side", data=np.array(colors_side, dtype=np.uint8), compression="gzip")
                    f.create_dataset("color_frames_concate", data=np.array(colors_concate, dtype=np.uint8), compression="gzip")
                    
                    # Depth frames
                    depths_front = buffer.get('depths_front')
                    depths_side = buffer.get('depths_side')
                    if depths_front is not None and len(depths_front) > 0:
                        f.create_dataset("depth_frames_front", data=np.array(depths_front, dtype=np.uint16), compression="gzip")
                    if depths_side is not None and len(depths_side) > 0:
                        f.create_dataset("depth_frames_side", data=np.array(depths_side, dtype=np.uint16), compression="gzip")
                    
                    # For backward compatibility, also save concatenated as color_frames
                    f.create_dataset("color_frames", data=np.array(colors_concate, dtype=np.uint8), compression="gzip")
                else:
                    # Old single-camera format (backward compatibility)
                    f.create_dataset("color_frames", data=np.array(colors, dtype=np.uint8), compression="gzip")
                    depths = buffer.get('depths')
                    if depths is not None and len(depths) > 0:
                        f.create_dataset("depth_frames", data=np.array(depths, dtype=np.uint16), compression="gzip")

                # 2. Robot State
                f.create_dataset("motions", data=np.array(buffer.get('joints', []), dtype=np.float32))
                f.create_dataset("ee_position", data=ee_position.astype(np.float32))
                f.create_dataset("ee_pose_quat", data=ee_pose_quat.astype(np.float32))
                f.create_dataset("gripper", data=np.array(buffer.get('grippers', []), dtype=np.int8))
                
                # 3. Timestamps
                f.create_dataset("timestamps", data=np.array(buffer.get('timestamps', []), dtype=np.float64))

                # 4. Task Information (Task Descriptions)
                if task_descriptions is None:
                    task_descriptions = [self.task_name]
                f.create_dataset("task_descriptions", 
                                 data=np.array(task_descriptions, dtype=h5py.string_dtype(encoding="utf-8")))
                f.create_dataset("task_switch_frames", data=np.array(task_switch_frames, dtype=np.int32))

                # 5. Metadata Attributes
                f.attrs["description"] = self.description
                f.attrs["created_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.attrs["dual_camera"] = use_dual_camera  # Mark if dual camera format

            if use_dual_camera:
                log.success(f"Saved {len(colors_front)} frames (dual-camera) to {filename}")
            else:
                log.success(f"Saved {len(colors)} frames to {filename}")

        except Exception as e:
            log.error(f"Failed to save episode {episode_idx}: {e}")

# ================= Self-Test Code (Updated to match new format) =================
if __name__ == "__main__":
    print("\n--- Starting DataSaver Self-Test (Target Format) ---\n")
    
    saver = DataSaver(save_dir="./output", task_name="test_task_001", description="Test Description")

    # Simulate Data (Buffer must include timestamps and grippers)
    N = 10
    dummy_buffer = {
        'colors': np.random.randint(0, 255, size=(N, 480, 640, 3), dtype=np.uint8),
        'depths': np.random.randint(0, 1000, size=(N, 480, 640), dtype=np.uint16),
        'joints': np.random.rand(N, 6),
        'ee_poses': np.random.rand(N, 7), # 7-dim: pos(3) + quat(4)
        'grippers': np.ones(N),
        'timestamps': np.linspace(0, 1, N),
        'task_switch_frames': np.linspace(4,2)
    }

    # Simulate task description list
    dummy_instructions = ["Move to apple", "Grasp apple"]
    task_switch_frames = np.random.rand(4, 2)

    saver.save_episode(dummy_buffer, 
                       episode_idx=888, 
                       task_descriptions=dummy_instructions,
                       task_switch_frames=task_switch_frames,
                       expected_stages=4)
    print("\n[PASS] Test run complete. Check ./output_test for result.")