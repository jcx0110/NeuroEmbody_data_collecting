#!/usr/bin/env python3
import os
import sys
import cv2
import time
from pathlib import Path

# 添加项目根目录到路径
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data_collecting.utils.logger import Logger as log
from data_collecting.utils.config_loader import ConfigLoader
from data_collecting.devices.realsense import RealSenseCamera
# 导入机械臂归位函数
from scripts.robot_home import move_to_home 

def main():
    cfg_loader = ConfigLoader()
    # 修正点：使用 cfg_loader.hardware 而不是 .config
    hw_cfg = cfg_loader.hardware 
    
    # 1. 机械臂归位
    robot_ip = hw_cfg.get('robot', {}).get('ip', '192.168.1.2')
    log.info(f"Step 1: Moving robot to home position at {robot_ip}...")
    if not move_to_home(robot_ip):
        log.error("Robot home failed. Aborting image capture.")
        return

    # 2. 初始化相机
    cameras = {}
    cam_configs = hw_cfg.get('cameras', {})
    
    for name in ['front', 'side']:
        if name not in cam_configs:
            log.warn(f"Camera '{name}' not found in hardware.yaml, skipping.")
            continue
            
        cfg = cam_configs[name]
        log.info(f"Initializing {name} camera (ID: {cfg['device_id']})...")
        cameras[name] = RealSenseCamera(
            device_id=cfg['device_id'],
            width=cfg.get('width', 640),
            height=cfg.get('height', 480),
            fps=cfg.get('fps', 30)
        )

    # 3. 拍摄并保存
    # 路径：NeuroEmbody_data_collecting/data_collecting/devices/
    save_path = root_dir / "data_collecting" / "devices"
    save_path.mkdir(parents=True, exist_ok=True)

    log.info("Capturing images...")
    time.sleep(1) # 给相机一点时间稳定

    try:
        for name, cam in cameras.items():
            color, _ = cam.read()
            if color is not None:
                file_path = save_path / f"standard_{name}.png"
                cv2.imwrite(str(file_path), color)
                log.success(f"Saved standard image for {name} to: {file_path}")
            else:
                log.error(f"Failed to capture frame from {name} camera.")
    finally:
        for cam in cameras.values():
            cam.stop()

if __name__ == "__main__":
    main()