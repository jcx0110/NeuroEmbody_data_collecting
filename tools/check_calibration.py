#!/usr/bin/env python3
import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import pyrealsense2 as rs
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from data_collecting.utils.logger import Logger as log
from data_collecting.utils.config_loader import ConfigLoader
from data_collecting.devices.realsense import RealSenseCamera

class PoseValidator:
    def __init__(self, marker_size=0.05):
        self.marker_size = marker_size
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.params = aruco.DetectorParameters()
        self.params.adaptiveThreshConstant = 7  # 默认值，可微调
        self.params.minMarkerPerimeterRate = 0.01 # 允许检测更小的码
        # 允许四边形顶点偏离理想位置的程度（默认 0.05，增大它可以识别更扁的码）
        # self.params.polygonalApproxAccuracyRate = 0.1 

        # 允许检测到的四边形长宽比有更大的偏差（默认是符合正方形）
        # 在新版本 OpenCV 中，可以调整这个参数来适应大角度
        self.params.minCornerDistanceRate = 0.1
        
        # 定义 3D 对象坐标：二维码的四个角在物体坐标系中的位置 (Z=0)
        # 顺序：左上, 右上, 右下, 左下 (ArUco 默认顺序)
        s = marker_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0],
            [ s,  s, 0],
            [ s, -s, 0],
            [-s, -s, 0]
        ], dtype=np.float32)

        # 兼容新版 OpenCV 检测器
        try:
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.params)
            self.is_new_api = True
        except AttributeError:
            self.is_new_api = False
            
        self.standard_poses = {} 

    def get_pose(self, img, K, D):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if self.is_new_api:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
            
        if ids is not None:
            curr_corners = corners[0][0].astype(np.float32)
            
            _, rvec, tvec = cv2.solvePnP(self.obj_points, curr_corners, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            aruco.drawDetectedMarkers(img, corners, ids) # 在原图画出检测到的码
            return rvec
        return None

    def calculate_angle_diff(self, rvec1, rvec2):
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        R_diff = np.dot(R1, R2.T)
        angle_diff_vec, _ = cv2.Rodrigues(R_diff)
        return np.degrees(np.linalg.norm(angle_diff_vec))

def main():
    cfg_loader = ConfigLoader()
    hw_cfg = cfg_loader.hardware # 修正点
    validator = PoseValidator(marker_size=0.05) 
    
    cams = {}
    intrinsic_data = {}
    std_path = root_dir / "data_collecting" / "devices"
    
    cam_configs = hw_cfg.get('cameras', {})

    for name in ['front', 'side']:
        if name not in cam_configs: continue
        
        cfg = cam_configs[name]
        cams[name] = RealSenseCamera(device_id=cfg['device_id'])
        
        # 从 RealSense Pipeline 获取内参
        profile = cams[name].profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = profile.get_intrinsics()
        K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
        D = np.array(intr.coeffs)
        intrinsic_data[name] = (K, D)

        # 加载刚才保存的标准图
        std_img = cv2.imread(str(std_path / f"standard_{name}.png"))
        if std_img is not None:
            rvec = validator.get_pose(std_img, K, D)
            if rvec is not None:
                validator.standard_poses[name] = rvec
                log.info(f"[{name}] Standard pose initialized.")
            else:
                log.warn(f"[{name}] ArUco marker NOT found in standard image!")
        else:
            log.error(f"[{name}] Standard image not found at {std_path}")

    log.info("Starting real-time check. Threshold: 1.5 deg. Press 'q' to exit.")
    
    try:
        while True:
            for name, cam in cams.items():
                img, _ = cam.read()
                if img is None: continue
                
                K, D = intrinsic_data[name]
                curr_rvec = validator.get_pose(img, K, D)
                
                if curr_rvec is not None and name in validator.standard_poses:
                    err = validator.calculate_angle_diff(curr_rvec, validator.standard_poses[name])
                    
                    # 设定阈值 1.5度
                    threshold = 1.5
                    color = (0, 255, 0) if err < threshold else (0, 0, 255)
                    status = "OK" if err < threshold else "MOVED"
                    
                    # 绘制坐标轴预览
                    cv2.drawFrameAxes(img, K, D, curr_rvec, np.zeros(3), 0.05)
                    cv2.putText(img, f"{name}: {err:.2f}deg ({status})", (20, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                cv2.imshow(f"Monitor - {name}", img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for cam in cams.values(): cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()