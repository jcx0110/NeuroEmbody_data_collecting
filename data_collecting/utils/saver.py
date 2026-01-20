import os
import sys
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path

# Path handling
try:
    from data_collecting.utils.logger import Logger as log
except ImportError:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from data_collecting.utils.logger import Logger as log

class DataSaver:
    def __init__(self, save_dir, task_name, description=""):
        self.task_name = task_name
        self.description = description
        self.output_dir = os.path.join(save_dir, task_name)
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            log.info(f"Initialized storage directory: {self.output_dir}")
        else:
            log.info(f"Storage directory exists: {self.output_dir}")

    def save_episode(self, buffer, episode_idx, task_descriptions=None, task_switch_frames=[], expected_stages=4):
        """
        Save a single episode to an HDF5 file with optimized structure.
        """
        # 1. Identify Data Availability
        colors_front = buffer.get('colors_front')
        colors_side = buffer.get('colors_side')
        colors_concate = buffer.get('colors_concate')
        
        if colors_front is None or (isinstance(colors_front, list) and len(colors_front) == 0):
            log.error("Essential image data (colors_front) is missing. Skipping save.")
            return False
            
        use_dual_camera = True # Logic assumes dual for this specific project iteration

        # 2. Stage Count Validation
        actual_stages = len(task_switch_frames)
        if actual_stages != expected_stages:
            log.error(f"VALIDATION FAILED: Stage count mismatch! (Expected {expected_stages}, Got {actual_stages})")
            log.error(f"DATA DISCARDED for Episode {episode_idx}")
            return False

        filename = f"episode_{episode_idx:03d}.h5"
        path = os.path.join(self.output_dir, filename)
        num_frames = len(colors_front)

        # 3. Process Kinematic Data (Focus on position and velocity)
        ee_poses = np.array(buffer.get('ee_poses', []))
        if len(ee_poses) > 0 and ee_poses.shape[1] == 7:
            ee_position = ee_poses[:, :3]
            ee_pose_quat = ee_poses 
        else:
            ee_position = np.zeros((num_frames, 3))
            ee_pose_quat = np.zeros((num_frames, 7))

        try:
            log.info(f"Starting saving Episode {episode_idx}...")
            
            # --- Pre-convert Large Arrays (Avoid blocking H5 write) ---
            log.info("Converting image arrays...")
            colors_front_arr = np.array(colors_front, dtype=np.uint8)
            colors_side_arr = np.array(colors_side, dtype=np.uint8)
            colors_concate_arr = np.array(colors_concate, dtype=np.uint8)
            
            depths_front_arr = np.array(buffer.get('depths_front', []), dtype=np.uint16)
            depths_side_arr = np.array(buffer.get('depths_side', []), dtype=np.uint16)

            # --- Process Motion Data (Action Space) ---
            # We explicitly ignore ee_velocity and ee_angular_velocity per user request.
            # We focus on joint_velocities for 'motions' dataset to ensure correct learning signals.
            joint_velocities_arr = np.array(buffer.get('joint_velocities', []), dtype=np.float32)
            
            if len(joint_velocities_arr) > 0:
                motions_data = joint_velocities_arr
            else:
                log.warn("joint_velocities not found, falling back to joint_positions for motions.")
                motions_data = np.array(buffer.get('joints', []), dtype=np.float32)

            log.info(f"Opening HDF5 file for writing: {filename}")
            with h5py.File(path, "w") as f:
                # --- 1. Efficient Image Storage ---
                # We save front/side for raw model input and concate for visualization.
                # 'color_frames' is DELETED to save space.
                img_grp = f.create_group("images")
                img_grp.create_dataset("front", data=colors_front_arr, compression="gzip", compression_opts=4)
                img_grp.create_dataset("side", data=colors_side_arr, compression="gzip", compression_opts=4)
                
                # We keep 'color_frames_concate' at root for backward compatibility with visualize_episode.py
                f.create_dataset("color_frames_concate", data=colors_concate_arr, compression="gzip", compression_opts=4)
                
                if len(depths_front_arr) > 0:
                    f.create_dataset("depth_frames_front", data=depths_front_arr, compression="gzip", compression_opts=4)
                if len(depths_side_arr) > 0:
                    f.create_dataset("depth_frames_side", data=depths_side_arr, compression="gzip", compression_opts=4)

                # --- 2. Robot State (Optimized) ---
                f.create_dataset("motions", data=motions_data)
                f.create_dataset("ee_position", data=ee_position.astype(np.float32))
                f.create_dataset("ee_pose_quat", data=ee_pose_quat.astype(np.float32))
                
                # Standard reference datasets
                f.create_dataset("joint_positions", data=np.array(buffer.get('joints', []), dtype=np.float32))
                f.create_dataset("joint_velocities", data=joint_velocities_arr)
                f.create_dataset("gripper", data=np.array(buffer.get('grippers', []), dtype=np.float32))
                f.create_dataset("timestamps", data=np.array(buffer.get('timestamps', []), dtype=np.float64))

                # --- 3. Task & Metadata ---
                if task_descriptions is None:
                    task_descriptions = [self.task_name]
                f.create_dataset("task_descriptions", 
                                 data=np.array(task_descriptions, dtype=h5py.string_dtype(encoding="utf-8")))
                f.create_dataset("task_switch_frames", data=np.array(task_switch_frames, dtype=np.int32))

                f.attrs["description"] = self.description
                f.attrs["created_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.attrs["dual_camera"] = True
                f.attrs["optimized_storage"] = True # Flag for cleaner structure

            log.success(f"Saved Episode {episode_idx} successfully. Frames: {num_frames}")
            return True

        except Exception as e:
            log.error(f"Failed to save episode {episode_idx}: {e}")
            return False

if __name__ == "__main__":
    # Minimal self-test logic here
    pass