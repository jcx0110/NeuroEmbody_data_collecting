#!/home/chenxi/anaconda3/envs/NeuroEmbody/bin/python
import h5py
import cv2
import numpy as np
import argparse
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from datetime import datetime
try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="husl")
    USE_SEABORN = True
except ImportError:
    USE_SEABORN = False

def check_display_available():
    """
    Check if X11 display is available for GUI visualization.
    """
    if os.environ.get('DISPLAY') is None:
        return False
    try:
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('test')
        return True
    except:
        return False

def process_depth_frame(depth_frame, depth_min=None, depth_max=None):
    """
    Normalize and colormap depth frames for visualization.
    """
    if depth_frame is None:
        return None
    
    depth_frame = depth_frame.copy()
    if depth_min is None or depth_max is None:
        valid_mask = (depth_frame > 0) & (depth_frame < 10000)
        if np.any(valid_mask):
            frame_min, frame_max = np.min(depth_frame[valid_mask]), min(np.max(depth_frame[valid_mask]), 10000)
        else:
            frame_min, frame_max = 300, 3000
    else:
        frame_min, frame_max = depth_min, depth_max
    
    depth_clipped = np.clip(depth_frame, frame_min, frame_max)
    depth_normalized = cv2.normalize(depth_clipped.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    invalid_mask = (depth_frame == 0) | (depth_frame >= 10000)
    depth_normalized[invalid_mask] = 0
    return cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

def process_frame(rgb_frame, depth_frame_front=None, depth_frame_side=None, idx=0, num_frames=0, 
                  status_text="", show_labels=True, depth_min=None, depth_max=None):
    """
    Combine RGB and depth streams into a single frame for video/GUI.
    """
    if rgb_frame.shape[-1] == 3:
        rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    else:
        rgb_bgr = rgb_frame
    
    depth_colored_front = process_depth_frame(depth_frame_front, depth_min, depth_max) if depth_frame_front is not None else None
    depth_colored_side = process_depth_frame(depth_frame_side, depth_min, depth_max) if depth_frame_side is not None else None
    
    if depth_colored_front is not None and depth_colored_side is not None:
        h_front, w_front = depth_colored_front.shape[:2]
        h_side, w_side = depth_colored_side.shape[:2]
        target_h = min(h_front, h_side)
        resized_front = cv2.resize(depth_colored_front, (int(w_front * (target_h/h_front)), target_h))
        resized_side = cv2.resize(depth_colored_side, (int(w_side * (target_h/h_side)), target_h))
        depth_combined = np.hstack([resized_front, resized_side])
        combined = np.hstack((rgb_bgr, depth_combined))
    elif depth_colored_front is not None:
        combined = np.hstack((rgb_bgr, depth_colored_front))
    else:
        combined = rgb_bgr
    
    if status_text:
        overlay = combined.copy()
        cv2.rectangle(overlay, (0, 0), (combined.shape[1], 40), (0, 0, 0), -1)
        combined = cv2.addWeighted(overlay, 0.7, combined, 0.3, 0)
        cv2.putText(combined, f"Frame: {idx+1}/{num_frames} [{status_text}]", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return combined

def generate_report(file_path, output_dir, data):
    """
    Generates a single comprehensive data report image containing:
    1. 3D Trajectory
    2. Joint Motions
    3. Statistics and Metadata
    4. Frame-by-frame Error Analysis (Velocity vs Motion)
    """
    # Extract data
    ee_pos = data['ee_position']
    motions = data['motions']
    timestamps = data['timestamps']
    ee_pose_quat = data.get('ee_pose_quat', None)  # [x, y, z, qx, qy, qz, qw] format
    time_axis = timestamps - timestamps[0] if timestamps is not None else np.arange(len(ee_pos))
    labels = ['X', 'Y', 'Z']  # Define labels early
    
    # Calculate velocity from position (for comparison)
    dt = np.diff(timestamps) if timestamps is not None else np.ones(len(ee_pos)-1) * (1.0/30.0)
    dt = np.where(dt > 0, dt, np.mean(dt[dt>0]) if np.any(dt>0) else 1.0/30.0)
    velocity_computed = np.diff(ee_pos, axis=0) / dt[:, np.newaxis]
    velocity_computed = np.vstack([velocity_computed, velocity_computed[-1:]]) # Pad to match length
    
    # Get measured velocities from motions dataset (first 3 dims should be EE velocity in m/s)
    # Also try to get from ee_velocity if available
    if 'ee_velocity' in data and data['ee_velocity'] is not None:
        velocity_measured = data['ee_velocity']  # Directly from RTDE
    elif motions.shape[1] >= 3:
        velocity_measured = motions[:, :3]  # From motions dataset
    else:
        velocity_measured = velocity_computed  # Fallback to computed
    
    # Calculate per-frame error: measured velocity vs computed velocity
    # This validates that RTDE measurements match position-derived velocity
    errors = velocity_measured - velocity_computed
    mse_per_dim = np.mean(errors**2, axis=0)
    total_mse = np.mean(errors**2)
    
    # Per-frame squared errors for detailed analysis
    per_frame_errors = np.sum(errors**2, axis=1)  # Sum of squared errors per frame
    
    # ========== Rotation Analysis (Quaternion-based) ==========
    angular_velocity_computed = None
    angular_velocity_measured = None
    rotation_angles = None
    rotation_errors = None
    angular_velocity_magnitude_computed = None
    angular_velocity_magnitude_measured = None
    
    if ee_pose_quat is not None and len(ee_pose_quat) > 0:
        # Format: [x, y, z, qx, qy, qz, qw] - extract quaternion part
        if len(ee_pose_quat.shape) == 2 and ee_pose_quat.shape[1] == 7:
            quaternions = ee_pose_quat[:, 3:7]  # Extract [qx, qy, qz, qw] from [x, y, z, qx, qy, qz, qw]
        else:
            # Invalid format, skip rotation analysis
            quaternions = None
        
        if quaternions is not None and len(quaternions) > 1 and not np.allclose(quaternions, 0.0, atol=1e-6):
            # Normalize quaternions
            quat_norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
            quaternions = quaternions / (quat_norms + 1e-10)  # Avoid division by zero
            
            # Calculate relative rotation between consecutive frames: Δq = q_current ⊗ q_previous^-1
            # rotation_angles[i] represents rotation from frame i-1 to frame i
            rotation_angles_list = []
            
            for i in range(1, len(quaternions)):
                q_prev = quaternions[i-1]  # [qx, qy, qz, qw] format - shape (4,)
                q_curr = quaternions[i]    # [qx, qy, qz, qw] format - shape (4,)
                
                # Convert to [w, x, y, z] format for easier calculation
                # q_prev/q_curr are in [qx, qy, qz, qw] format, so:
                w_prev, x_prev, y_prev, z_prev = q_prev[3], q_prev[0], q_prev[1], q_prev[2]
                w_curr, x_curr, y_curr, z_curr = q_curr[3], q_curr[0], q_curr[1], q_curr[2]
                
                # Quaternion inverse: q^-1 = [w, -x, -y, -z] for normalized quaternion
                # Δq = q_curr ⊗ q_prev^-1
                w_inv = w_prev
                x_inv = -x_prev
                y_inv = -y_prev
                z_inv = -z_prev
                
                # Quaternion multiplication: q1 ⊗ q2
                delta_w = w_curr*w_inv - x_curr*x_inv - y_curr*y_inv - z_curr*z_inv
                delta_x = w_curr*x_inv + x_curr*w_inv + y_curr*z_inv - z_curr*y_inv
                delta_y = w_curr*y_inv - x_curr*z_inv + y_curr*w_inv + z_curr*x_inv
                delta_z = w_curr*z_inv + x_curr*y_inv - y_curr*x_inv + z_curr*w_inv
                
                # Normalize delta quaternion
                delta_norm = np.sqrt(delta_w**2 + delta_x**2 + delta_y**2 + delta_z**2) + 1e-10
                delta_w /= delta_norm
                delta_x /= delta_norm
                delta_y /= delta_norm
                delta_z /= delta_norm
                
                # Calculate angular distance: Angle = 2 * arccos(|w|)
                w_abs = abs(delta_w)
                w_abs = np.clip(w_abs, -1.0, 1.0)  # Clamp to valid range for arccos
                angle = 2.0 * np.arccos(w_abs)
                rotation_angles_list.append(angle)
            
            # Convert to numpy array: rotation_angles[i] = angle from frame i-1 to frame i
            # So rotation_angles[0] = 0 (no previous frame), rotation_angles[1] = angle from frame 0 to 1
            rotation_angles = np.array([0.0] + rotation_angles_list)  # Shape: (N,)
            
            # Calculate angular velocity magnitude from rotation angles: |ω| = angle / dt
            # dt[i] = timestamps[i+1] - timestamps[i], so dt[i] corresponds to rotation from frame i to i+1
            # rotation_angles[i+1] = angle from frame i to i+1, so it should use dt[i]
            # Therefore: rotation_angles[0] uses dt[0] (or 0), rotation_angles[1] uses dt[0], rotation_angles[i] uses dt[i-1]
            if len(dt) == len(rotation_angles) - 1:
                # Perfect match: dt[i] corresponds to rotation_angles[i+1]
                # rotation_angles[0] = 0, so we use dt[0] for it (or keep it 0)
                # rotation_angles[1:] uses dt[0:]
                dt_rotation = np.concatenate([[dt[0] if len(dt) > 0 else 1.0/30.0], dt])
            elif len(dt) == len(rotation_angles):
                # dt already has same length (shouldn't happen normally, but handle it)
                dt_rotation = dt
            else:
                # Mismatch: use actual timestamps to compute dt for each rotation
                # Calculate dt for each rotation angle directly from timestamps
                if timestamps is not None and len(timestamps) == len(rotation_angles):
                    dt_rotation = np.diff(timestamps)
                    dt_rotation = np.concatenate([[dt_rotation[0] if len(dt_rotation) > 0 else 1.0/30.0], dt_rotation])
                else:
                    # Fallback: use mean dt for all frames
                    mean_dt = np.mean(dt) if len(dt) > 0 else 1.0/30.0
                    dt_rotation = np.full(len(rotation_angles), mean_dt)
            
            # Ensure dt_rotation is positive and matches rotation_angles length
            dt_rotation = np.where(dt_rotation > 0, dt_rotation, np.mean(dt_rotation[dt_rotation > 0]) if np.any(dt_rotation > 0) else 1.0/30.0)
            if len(dt_rotation) != len(rotation_angles):
                # Final fallback: pad or truncate to match
                if len(dt_rotation) < len(rotation_angles):
                    mean_dt = np.mean(dt_rotation) if len(dt_rotation) > 0 else 1.0/30.0
                    dt_rotation = np.concatenate([dt_rotation, np.full(len(rotation_angles) - len(dt_rotation), mean_dt)])
                else:
                    dt_rotation = dt_rotation[:len(rotation_angles)]
            
            angular_velocity_magnitude_computed = rotation_angles / (dt_rotation + 1e-10)  # rad/s
            
            # Get measured angular velocity from RTDE if available
            if 'ee_angular_velocity' in data and data['ee_angular_velocity'] is not None:
                angular_velocity_measured = data['ee_angular_velocity']  # [wx, wy, wz] in rad/s
                angular_velocity_magnitude_measured = np.linalg.norm(angular_velocity_measured, axis=1)  # |ω| in rad/s
                
                # Ensure lengths match
                if len(angular_velocity_magnitude_measured) == len(angular_velocity_magnitude_computed):
                    # Calculate error: measured vs computed magnitude
                    rotation_errors = angular_velocity_magnitude_measured - angular_velocity_magnitude_computed
                else:
                    rotation_errors = None
            else:
                angular_velocity_measured = None
                angular_velocity_magnitude_measured = None
                rotation_errors = None
        else:
            # No valid quaternion data
            rotation_angles = np.zeros(len(ee_pos))
            angular_velocity_magnitude_computed = np.zeros(len(ee_pos))
            angular_velocity_magnitude_measured = None
            rotation_errors = None
    else:
        rotation_angles = np.zeros(len(ee_pos))
        angular_velocity_magnitude_computed = np.zeros(len(ee_pos))
        angular_velocity_magnitude_measured = None
        rotation_errors = None
    
    # Setup Figure with GridSpec - Non-uniform layout with large top section
    # Layout: 5 rows x 4 columns (Top section 2.5x height + Bottom compact 3x3 grid + Bottom text row)
    fig = plt.figure(figsize=(26, 24), facecolor='#f8f9fa')
    gs = GridSpec(5, 4, figure=fig, 
                  height_ratios=[2.5, 1.0, 1.0, 1.0, 0.6], 
                  width_ratios=[1.2, 1.0, 1.0, 0.8],
                  hspace=0.5, wspace=0.4)
    
    # Modern color palette
    colors = {
        'primary': '#2E86AB',      # Blue
        'secondary': '#A23B72',   # Purple
        'accent': '#F18F01',      # Orange
        'success': '#06A77D',     # Green
        'warning': '#F77F00',     # Dark Orange
        'error': '#D62828',       # Red
        'text': '#2C3E50',        # Dark Gray
        'bg': '#FFFFFF',
        'grid': '#E8E8E8'
    }
    
    # Header Information with modern styling
    plt.suptitle(f"NEUROEMBODY EPISODE REPORT", fontsize=34, fontweight='bold', 
                color=colors['text'], y=0.98)
    plt.figtext(0.5, 0.97, file_path.name, fontsize=20, ha='center', 
               style='italic', color='#7F8C8D', alpha=0.8)
    
    # 1. 3D Trajectory (Top Left - Large, spans 2 columns) - Modern styling
    ax_3d = fig.add_subplot(gs[0, 0:2], projection='3d', facecolor='white')
    ax_3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], 
              color=colors['primary'], lw=3, alpha=0.85, label='Path', zorder=1)
    ax_3d.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], 
                 color=colors['success'], s=200, label='Start', zorder=3, edgecolors='white', linewidths=2.5)
    ax_3d.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], 
                 color=colors['error'], s=200, label='End', zorder=3, edgecolors='white', linewidths=2.5)
    ax_3d.set_title("End-Effector 3D Trajectory", fontsize=20, fontweight='bold', 
                   pad=20, color=colors['text'])
    ax_3d.set_xlabel("X (m)", fontsize=13, fontweight='medium')
    ax_3d.set_ylabel("Y (m)", fontsize=13, fontweight='medium')
    ax_3d.set_zlabel("Z (m)", fontsize=13, fontweight='medium')
    ax_3d.grid(True, alpha=0.25, color=colors['grid'])
    ax_3d.legend(loc='upper left', framealpha=0.95, fontsize=11, fancybox=True, shadow=True)

    # 2. Velocities Overview (Top Right - Compact) - Modern styling
    ax_joints = fig.add_subplot(gs[0, 2:4], facecolor='white')
    # Plot EE velocities (first 3 dims) and joint velocities (remaining dims)
    ee_colors = [colors['primary'], colors['secondary'], colors['accent']]
    if motions.shape[1] >= 3:
        # EE velocities (m/s) - solid lines
        for i in range(3):
            ax_joints.plot(time_axis, motions[:, i], color=ee_colors[i], alpha=0.85, 
                         linestyle='-', lw=2.5, label=f'EE Vel {labels[i]}', zorder=3)
        # Joint velocities (rad/s) if available - dashed lines
        if motions.shape[1] > 3:
            joint_colors = plt.cm.viridis(np.linspace(0.2, 0.8, motions.shape[1] - 3))
            for i in range(3, motions.shape[1]):
                ax_joints.plot(time_axis, motions[:, i], color=joint_colors[i-3], alpha=0.65, 
                             linestyle='--', lw=1.8, label=f'Joint {i-2} Vel', zorder=2)
    else:
        # Fallback: plot all as joint velocities
        joint_colors = plt.cm.viridis(np.linspace(0.2, 0.8, motions.shape[1]))
        for i in range(motions.shape[1]):
            ax_joints.plot(time_axis, motions[:, i], color=joint_colors[i], alpha=0.8, 
                         lw=2, label=f'J{i+1}')
    ax_grip = ax_joints.twinx()  # Create twin axis for different scale
    ax_grip.plot(time_axis, data.get('gripper', np.zeros_like(time_axis)), color='m', lw=2, label='Gripper')
    ax_grip.set_ylabel('Gripper State', color='m') # Add label
    ax_grip.grid(False) # Disable secondary grid to keep it clean
    ax_joints.set_title("Velocities Overview (EE: m/s, Joints: rad/s)", fontsize=18, 
                       fontweight='bold', pad=15, color=colors['text'])
    ax_joints.set_xlabel("Time (s)", fontsize=12, fontweight='medium')
    ax_joints.set_ylabel("Velocity", fontsize=12, fontweight='medium')
    ax_joints.grid(True, alpha=0.25, color=colors['grid'], linestyle='-', linewidth=0.8)
    ax_joints.legend(loc='upper right', ncol=2, fontsize=9, framealpha=0.95, 
                    fancybox=True, shadow=True)
    ax_joints.spines['top'].set_visible(False)
    ax_joints.spines['right'].set_visible(False)

    # 3. Compact Analysis Grid (3x3) - Bottom section
    # Row 1: Velocity Validation (X, Y, Z) - columns 1, 2, 3
    for i in range(3):
        ax_v = fig.add_subplot(gs[1, i+1], facecolor='white')
        ax_v.plot(time_axis, velocity_computed[:, i], color=colors['primary'], lw=2, 
                 label='Computed', alpha=0.8, zorder=3)
        ax_v.plot(time_axis, velocity_measured[:, i], color=colors['error'], linestyle='--', 
                 lw=2, label='Measured', alpha=0.8, zorder=3)
        ax_v.set_title(f"Vel Validation: {labels[i]}", fontsize=12, fontweight='bold', 
                      pad=8, color=colors['text'])
        ax_v.set_ylabel("Vel (m/s)", fontsize=10, fontweight='medium')
        # Hide xlabel for middle row
        ax_v.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.6)
        ax_v.spines['top'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        ax_v.tick_params(labelsize=8)
        if i == 0: 
            ax_v.legend(loc='upper left', framealpha=0.95, fancybox=True, shadow=True, fontsize=8)

    # Row 2: Linear Velocity Errors (X, Y, Z) - columns 1, 2, 3
    for i in range(3):
        ax_e = fig.add_subplot(gs[2, i+1], facecolor='white')
        # Plot per-frame error with gradient fill
        ax_e.fill_between(time_axis, errors[:, i], 0, color=colors['accent'], 
                         alpha=0.3, label='Error', zorder=1)
        ax_e.plot(time_axis, errors[:, i], color=colors['warning'], lw=1.8, alpha=0.9, zorder=2)
        ax_e.axhline(0, color=colors['text'], lw=1, linestyle='--', alpha=0.5, zorder=1)
        # Add per-frame squared error as secondary y-axis
        ax_e2 = ax_e.twinx()
        ax_e2.plot(time_axis, errors[:, i]**2, color=colors['secondary'], lw=1.5, 
                  alpha=0.6, label='Sq Error', zorder=2)
        ax_e2.set_ylabel("Sq Error", color=colors['secondary'], fontsize=10, fontweight='medium')
        ax_e2.tick_params(axis='y', labelcolor=colors['secondary'], labelsize=9)
        ax_e.set_title(f"Lin Vel Error {labels[i]} (MSE: {mse_per_dim[i]:.4f})", 
                      fontsize=12, fontweight='bold', pad=10, color=colors['text'])
        # Hide xlabel for middle row
        ax_e.set_ylabel("Error (m/s)", color=colors['warning'], fontsize=10, fontweight='medium')
        ax_e.tick_params(axis='y', labelcolor=colors['warning'], labelsize=9)
        ax_e.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.6)
        ax_e.spines['top'].set_visible(False)
        ax_e.spines['right'].set_visible(False)
    
    # Row 3: Rotation Analysis (Angle, Angular Vel, Angular Vel Error) - columns 1, 2, 3
    if rotation_angles is not None and angular_velocity_magnitude_computed is not None:
        # Left: Rotation angles over time
        ax_rot1 = fig.add_subplot(gs[3, 1], facecolor='white')
        ax_rot1.plot(time_axis, np.degrees(rotation_angles), color=colors['success'], 
                    lw=2, label='Angle', alpha=0.85, zorder=3)
        ax_rot1.set_title("Rotation Angle", fontsize=12, fontweight='bold', 
                         pad=10, color=colors['text'])
        ax_rot1.set_xlabel("Time (s)", fontsize=11, fontweight='medium')
        ax_rot1.set_ylabel("Angle (deg)", fontsize=10, fontweight='medium')
        ax_rot1.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.6)
        ax_rot1.tick_params(labelsize=9)
        ax_rot1.spines['top'].set_visible(False)
        ax_rot1.spines['right'].set_visible(False)
        
        # Middle: Angular velocity comparison
        ax_rot2 = fig.add_subplot(gs[3, 2], facecolor='white')
        ax_rot2.plot(time_axis, angular_velocity_magnitude_computed, color=colors['primary'], 
                    lw=2, label='Computed', alpha=0.85, zorder=3)
        if angular_velocity_magnitude_measured is not None:
            ax_rot2.plot(time_axis, angular_velocity_magnitude_measured, color=colors['error'], 
                        linestyle='--', lw=2, label='Measured', alpha=0.85, zorder=3)
        ax_rot2.set_title("Angular Vel Validation", fontsize=12, fontweight='bold', 
                         pad=10, color=colors['text'])
        ax_rot2.set_xlabel("Time (s)", fontsize=11, fontweight='medium')
        ax_rot2.set_ylabel("Ang Vel (rad/s)", fontsize=10, fontweight='medium')
        ax_rot2.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.6)
        ax_rot2.tick_params(labelsize=9)
        ax_rot2.legend(framealpha=0.95, fancybox=True, shadow=True, fontsize=9, loc='upper left')
        ax_rot2.spines['top'].set_visible(False)
        ax_rot2.spines['right'].set_visible(False)
        
        # Right: Angular velocity error (if measured available)
        ax_rot3 = fig.add_subplot(gs[3, 3], facecolor='white')
        if rotation_errors is not None:
            ax_rot3.fill_between(time_axis, rotation_errors, 0, color=colors['secondary'], 
                               alpha=0.3, label='Error', zorder=1)
            ax_rot3.plot(time_axis, rotation_errors, color=colors['secondary'], lw=1.8, alpha=0.9, zorder=2)
            ax_rot3.axhline(0, color=colors['text'], lw=1, linestyle='--', alpha=0.5, zorder=1)
            rot_mse = np.mean(rotation_errors**2)
            ax_rot3.set_title(f"Ang Vel Error (MSE: {rot_mse:.4f})", fontsize=12, 
                            fontweight='bold', pad=10, color=colors['text'])
            ax_rot3.set_ylabel("Error (rad/s)", color=colors['secondary'], fontsize=10, fontweight='medium')
            ax_rot3.tick_params(axis='y', labelcolor=colors['secondary'], labelsize=9)
        else:
            ax_rot3.text(0.5, 0.5, 'No RTDE angular\nvelocity data', 
                        ha='center', va='center', transform=ax_rot3.transAxes, fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_rot3.set_title("Angular Vel Error", fontsize=12, fontweight='bold', 
                            pad=10, color=colors['text'])
        ax_rot3.set_xlabel("Time (s)", fontsize=11, fontweight='medium')
        ax_rot3.grid(True, alpha=0.2, color=colors['grid'], linestyle='-', linewidth=0.6)
        ax_rot3.tick_params(axis='x', labelsize=9)
        ax_rot3.spines['top'].set_visible(False)
        ax_rot3.spines['right'].set_visible(False)
    else:
        # No rotation data available
        for i in range(1, 4):
            ax_rot = fig.add_subplot(gs[3, i], facecolor='white')
            ax_rot.text(0.5, 0.5, 'No rotation data available', 
                        ha='center', va='center', transform=ax_rot.transAxes, fontsize=10)
            ax_rot.set_title("Rotation Analysis", fontsize=11, fontweight='bold', 
                            pad=8, color=colors['text'])
            ax_rot.axis('off')
    
    # 4. Summary Statistics Box (Left side, below 3D trajectory)
    max_per_frame_error = np.max(per_frame_errors)
    mean_per_frame_error = np.mean(per_frame_errors)
    
    # Rotation statistics
    rot_stats = ""
    if rotation_angles is not None and len(rotation_angles) > 0:
        max_rot_angle = np.max(rotation_angles)
        mean_rot_angle = np.mean(rotation_angles)
        rot_stats = (
            f"\n╔═══════════════════════════════════╗\n"
            f"║ ROTATION ANALYSIS                 ║\n"
            f"╠═══════════════════════════════════╣\n"
            f"║ Max Angle: {np.degrees(max_rot_angle):6.3f}°              ║\n"
            f"║ Mean Angle: {np.degrees(mean_rot_angle):5.3f}°              ║\n"
        )
        if rotation_errors is not None:
            rot_mse = np.mean(rotation_errors**2)
            rot_stats += (
                f"║ Ang Vel MSE: {rot_mse:8.6f}          ║\n"
                f"║ Max Error: {np.max(np.abs(rotation_errors)):9.6f}          ║\n"
            )
        rot_stats += f"╚═══════════════════════════════════╝\n"
    
    stats_text = (
        f"╔═══════════════════════════════════╗\n"
        f"║ DATA SUMMARY                     ║\n"
        f"╠═══════════════════════════════════╣\n"
        f"║ Frames: {len(ee_pos):3d}                        ║\n"
        f"║ Duration: {time_axis[-1]:5.2f}s                  ║\n"
        f"║ Avg FPS: {len(ee_pos)/time_axis[-1]:4.1f}                     ║\n"
        f"╚═══════════════════════════════════╝\n\n"
        f"╔═══════════════════════════════════╗\n"
        f"║ LINEAR VELOCITY                  ║\n"
        f"╠═══════════════════════════════════╣\n"
        f"║ Total MSE: {total_mse:8.6f}          ║\n"
        f"║ X-MSE: {mse_per_dim[0]:11.6f}          ║\n"
        f"║ Y-MSE: {mse_per_dim[1]:11.6f}          ║\n"
        f"║ Z-MSE: {mse_per_dim[2]:11.6f}          ║\n"
        f"╚═══════════════════════════════════╝\n\n"
        f"╔═══════════════════════════════════╗\n"
        f"║ PER-FRAME ERRORS                 ║\n"
        f"╠═══════════════════════════════════╣\n"
        f"║ Max: {max_per_frame_error:11.6f}          ║\n"
        f"║ Mean: {mean_per_frame_error:10.6f}          ║\n"
        f"║ Std: {np.std(per_frame_errors):12.6f}          ║\n"
        f"╚═══════════════════════════════════╝"
        f"{rot_stats}"
        f"╔═══════════════════════════════════╗\n"
        f"║ TIMESTAMP                        ║\n"
        f"╠═══════════════════════════════════╣\n"
        f"║ {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):27s} ║\n"
        f"╚═══════════════════════════════════╝"
    )
    # Position statistics box on left side, below 3D trajectory (in column 0, rows 1-3)
    # Use a dedicated subplot area for statistics to avoid overlap
    ax_stats = fig.add_subplot(gs[1:4, 0])
    ax_stats.axis('off')
    ax_stats.text(0.05, 0.98, stats_text, transform=ax_stats.transAxes, 
                 fontsize=12, fontfamily='monospace', 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=2', facecolor='white', alpha=0.98, 
                          edgecolor=colors['primary'], linewidth=2.5, pad=15),
                 color=colors['text'])
    
    # 5. H5 File Keys Information - Place on the right side, below velocities overview
    keys_info = data.get('keys_info', {})
    attrs_info = data.get('attrs_info', {})
    
    if keys_info:
        # Create a more readable format with better spacing and wider layout
        keys_text = "H5 FILE STRUCTURE\n"
        keys_text += "═" * 110 + "\n\n"
        keys_text += "DATASETS:\n"
        keys_text += "─" * 110 + "\n"
        
        # Sort keys for better readability (group by type)
        image_keys = [k for k in keys_info.keys() if 'color' in k.lower() or 'depth' in k.lower() or 'rgb' in k.lower()]
        robot_keys = [k for k in keys_info.keys() if k not in image_keys]
        
        # Display image keys first (limit to 3 most important)
        for key in sorted(image_keys)[:3]:
            info = keys_info[key]
            shape_str = "x".join(map(str, info['shape']))
            size_str = f"{info['size_mb']:.2f} MB" if info['size_mb'] > 0.01 else f"{info['size_mb']*1024:.2f} KB"
            keys_text += f"  • {key:40s} | Shape: {shape_str:30s} | Size: {size_str:15s}\n"
        if len(image_keys) > 3:
            keys_text += f"  ... and {len(image_keys)-3} more image datasets\n"
        
        if image_keys and robot_keys:
            keys_text += "\n"
        
        # Display robot state keys
        for key in sorted(robot_keys):
            info = keys_info[key]
            shape_str = "x".join(map(str, info['shape']))
            size_str = f"{info['size_mb']:.2f} MB" if info['size_mb'] > 0.01 else f"{info['size_mb']*1024:.2f} KB"
            desc = info['description'][:35] if len(info['description']) > 35 else info['description']
            keys_text += f"  • {key:40s} | Shape: {shape_str:30s} | {desc:35s} | {size_str:15s}\n"
        
        if attrs_info:
            keys_text += "\nATTRIBUTES:\n"
            keys_text += "─" * 110 + "\n"
            for key, value in sorted(attrs_info.items()):
                # Truncate long values
                value_str = str(value)
                if len(value_str) > 70:
                    value_str = value_str[:67] + "..."
                keys_text += f"  • {key:40s} | {value_str}\n"
        
        # Place H5 keys info in a dedicated subplot area (bottom row, centered, spans columns 0-3)
        ax_keys = fig.add_subplot(gs[4, :])
        ax_keys.axis('off')
        # Center the text box horizontally
        ax_keys.text(0.5, 0.98, keys_text, transform=ax_keys.transAxes, 
                    fontsize=14, fontfamily='monospace', 
                    verticalalignment='top', horizontalalignment='center',
                    bbox=dict(boxstyle='round,pad=3', facecolor='white', alpha=0.98, 
                            edgecolor=colors['secondary'], linewidth=3, pad=20),
                    color=colors['text'])

    report_path = output_dir / f"{file_path.stem}_comprehensive_report.png"
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comprehensive report saved to: {report_path}")

def get_h5_keys_info(file_path):
    """
    Extract all keys from H5 file and provide descriptions.
    """
    keys_info = {}
    
    with h5py.File(file_path, 'r') as f:
        # Get all dataset keys
        def get_key_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape
                dtype = obj.dtype
                size_mb = obj.nbytes / (1024 * 1024)
                
                # Provide description based on key name
                description = ""
                if 'color' in name.lower() or 'rgb' in name.lower():
                    description = "RGB image frames"
                elif 'depth' in name.lower():
                    description = "Depth image frames"
                elif 'motion' in name.lower():
                    description = "Joint/EE velocities (m/s, rad/s)"
                elif 'position' in name.lower() and 'ee' in name.lower():
                    description = "End-effector position (m)"
                elif 'pose' in name.lower() and 'quat' in name.lower():
                    description = "EE pose [x,y,z,qx,qy,qz,qw]"
                elif 'velocity' in name.lower() and 'angular' in name.lower():
                    description = "EE angular velocity [wx,wy,wz] (rad/s)"
                elif 'velocity' in name.lower() and 'ee' in name.lower():
                    description = "EE linear velocity [vx,vy,vz] (m/s)"
                elif 'joint' in name.lower() and 'velocit' in name.lower():
                    description = "Joint velocities (rad/s)"
                elif 'joint' in name.lower() and 'position' in name.lower():
                    description = "Joint positions (rad)"
                elif 'gripper' in name.lower():
                    description = "Gripper state"
                elif 'timestamp' in name.lower():
                    description = "Frame timestamps (s)"
                elif 'task' in name.lower():
                    description = "Task information"
                else:
                    description = "Data array"
                
                keys_info[name] = {
                    'shape': shape,
                    'dtype': str(dtype),
                    'size_mb': size_mb,
                    'description': description
                }
        
        f.visititems(get_key_info)
        
        # Also get attributes
        attrs_info = {}
        for key in f.attrs.keys():
            attrs_info[key] = str(f.attrs[key])
    
    return keys_info, attrs_info

def analyze_h5_data(file_path):
    """
    Main analysis function to load data and call report generation.
    """
    file_path = Path(file_path)
    output_dir = file_path.parent
    
    # Get H5 file keys information
    keys_info, attrs_info = get_h5_keys_info(file_path)
    
    with h5py.File(file_path, 'r') as f:
        data = {
            'ee_position': f['ee_position'][:] if 'ee_position' in f else None,
            'motions': f['motions'][:] if 'motions' in f else None,
            'timestamps': f['timestamps'][:] if 'timestamps' in f else None,
            'gripper': f['gripper'][:] if 'gripper' in f else None,
            'ee_velocity': f['ee_velocity'][:] if 'ee_velocity' in f else None,  # RTDE measured linear velocity
            'ee_angular_velocity': f['ee_angular_velocity'][:] if 'ee_angular_velocity' in f else None,  # RTDE measured angular velocity
            'ee_pose_quat': f['ee_pose_quat'][:] if 'ee_pose_quat' in f else None,  # Quaternion [x, y, z, qx, qy, qz, qw]
            'keys_info': keys_info,  # Add keys info to data
            'attrs_info': attrs_info  # Add attributes info
        }
        
        if data['ee_position'] is not None and data['motions'] is not None:
            if not np.all(data['ee_position'] == 0):
                generate_report(file_path, output_dir, data)
            else:
                print("Warning: All end-effector positions are zero. Report skipped.")

def visualize_episode(file_path, force_save_video=False, fps=30.0):
    """
    Main entry point for visualization and analysis.
    """
    file_path = Path(file_path)
    if not file_path.exists(): return
    
    print(f"Processing episode: {file_path.name}")
    analyze_h5_data(file_path)
    
    try:
        with h5py.File(file_path, 'r') as f:
            is_dual = f.attrs.get('dual_camera', False)
            rgb_frames = f['color_frames_concate'][:] if is_dual and 'color_frames_concate' in f else f['color_frames'][:]
            depth_f = f['depth_frames_front'][:] if 'depth_frames_front' in f else f.get('depth_frames')
            depth_s = f.get('depth_frames_side')
            
            if force_save_video or not check_display_available():
                output_path = file_path.parent / f"{file_path.stem}_visualization.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                test_img = process_frame(rgb_frames[0], depth_f[0] if depth_f is not None else None)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (test_img.shape[1], test_img.shape[0]))
                
                for i in range(len(rgb_frames)):
                    frame = process_frame(rgb_frames[i], 
                                         depth_f[i] if depth_f is not None else None,
                                         depth_s[i] if depth_s is not None else None, 
                                         i, len(rgb_frames), "EXPORTING")
                    out.write(frame)
                out.release()
                print(f"Video saved: {output_path}")
    except Exception as e:
        print(f"Error during visualization: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroEmbody Data Visualization and Reporting Tool")
    parser.add_argument("file_path", type=str, help="Path to h5 file")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    
    visualize_episode(args.file_path, force_save_video=args.save_video, fps=args.fps)