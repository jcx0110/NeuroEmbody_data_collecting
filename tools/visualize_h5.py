import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

def visualize_h5(file_path):
    """
    Reads an .h5 file and visualizes video + robot state data.
    Syncs with multi-stage task descriptions.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[Error] File not found: {file_path}")
        return

    print(f"Opening: {file_path.name}...")

    with h5py.File(file_path, 'r') as f:
        # 1. Read Metadata
        print("="*40)
        print(" METADATA & TASK INFO")
        print("="*40)
        for key, val in f.attrs.items():
            print(f"{key}: {val}")
        
        # 2. Read Datasets
        rgb_frames = f['color_frames'][:]
        depth_frames = f['depth_frames'][:] if 'depth_frames' in f else None
        joint_positions = f['motions'][:]
        gripper_states = f['gripper'][:]
        
        # Load Task Stage Information
        # task_descriptions: List of strings (instructions)
        # task_switch_frames: Indices where each stage starts
        task_instructions = [t.decode('utf-8') for t in f['task_descriptions'][:]]
        task_switches = f['task_switch_frames'][:] 
        
        print(f"\nTask Sequence:")
        for i, (instr, start_frame) in enumerate(zip(task_instructions, task_switches)):
            # Handling both (N, 2) format and (N,) format for switches
            frame_val = start_frame[1] if isinstance(start_frame, (np.ndarray, list)) else start_frame
            print(f"  Stage {i+1} [Frame {frame_val}]: {instr}")

        # 3. Visualization Setup
        num_frames = len(rgb_frames)
        plot_joints(joint_positions, gripper_states)
        
        paused = False
        idx = 0
        cv2.namedWindow('NeuroEmbody Data Viewer', cv2.WINDOW_NORMAL)

        while idx < num_frames:
            if not paused:
                # RGB to BGR for OpenCV
                frame = cv2.cvtColor(rgb_frames[idx], cv2.COLOR_RGB2BGR)
                
                # Sync Task Instruction based on current frame index
                current_instr = "Unknown"
                # Find the last switch point that is <= current index
                for i, start_frame in enumerate(task_switches):
                    frame_val = start_frame[1] if isinstance(start_frame, (np.ndarray, list)) else start_frame
                    if idx >= frame_val:
                        current_instr = f"Stage {i+1}: {task_instructions[i]}"
                
                # Process Depth
                if depth_frames is not None:
                    depth_map = cv2.normalize(depth_frames[idx], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
                    canvas = np.hstack((frame, depth_map))
                else:
                    canvas = frame

                # UI Overlays
                # 1. Background Bar for Text
                cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 70), (0, 0, 0), -1)
                # 2. Frame Info
                cv2.putText(canvas, f"Frame: {idx}/{num_frames} | {'PAUSED' if paused else 'PLAYING'}", 
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                # 3. Active Instruction (Highlight in Green)
                cv2.putText(canvas, current_instr, (10, 55), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('NeuroEmbody Data Viewer', canvas)
                idx += 1
            
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused
                # Update UI to show paused state immediately
                continue 
            elif key == ord('a') and idx > 1: # Step back
                idx = max(0, idx - 2)
            elif key == ord('d') and idx < num_frames - 1: # Step forward
                idx = min(num_frames - 1, idx + 1)
        
        cv2.destroyAllWindows()
        plt.close('all')

def plot_joints(joints, grippers):
    """Plots joint positions and gripper states."""
    plt.ion() # Interaction mode on
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    
    for i in range(joints.shape[1]):
        ax1.plot(joints[:, i], label=f'J{i+1}')
    ax1.set_title("Robot Joint Motions")
    ax1.set_ylabel("Position (Rad/m)")
    ax1.legend(loc='right', fontsize='x-small')
    ax1.grid(alpha=0.3)

    ax2.step(range(len(grippers)), grippers, color='red', where='post', label='Gripper')
    ax2.set_title("Gripper State (0=Open, 1=Closed)")
    ax2.set_ylim(-0.2, 1.2)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="output/test_task_001/episode_888.h5", help="H5 file path")
    args = parser.parse_args()
    
    visualize_h5(args.path)