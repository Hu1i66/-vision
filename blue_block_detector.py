#!/home/lxf/orange_dataset/.venv/bin/python3
# -*- coding: utf-8 -*-
"""蓝方块视觉识别节点 (Task 1.1-1.4)

HSV 颜色分割 + cv2.minAreaRect 旋转矩形检测 + RANSAC 平面拟合深度估计 + 三维坐标发布。
架构与 realsense_yolo_node.py 一致 (节点结构/订阅发布/ObjectTrack EMA/眼在手外标定/坐标转换/JSON 发布)。

运行前提: source /opt/ros/humble/setup.bash (提供 rclpy / cv_bridge / sensor_msgs)
Python 环境: /home/lxf/orange_dataset/.venv (提供 open3d / cv2 / numpy)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge

import cv2
import numpy as np
import open3d as o3d
import json
import math
import subprocess


# ====================================================================
# ObjectTrack: 蓝方块单目标 EMA 滤波 (适配自 realsense_yolo_node.py)
# 对 bbox 中心 / 旋转角 / 宽高 / 面积做帧间平滑, α=0.25
# ====================================================================
class ObjectTrack:
    """单个蓝方块 track 的 EMA 滤波状态。

    平滑字段: 中心 (center_u, center_v)、旋转角 (angle, [0°,180°))、
              长边 s_w、短边 s_h、面积 s_area。
    旋转角在圆周 [0,180) 上做最短路径 EMA, 避免跨 0°/180° 跳变。
    """

    def __init__(self, track_id, obs, frame_count, alpha=0.25):
        self.track_id = track_id
        self.class_name = "blue_block"
        self.alpha = alpha  # EMA 系数: 新观测权重, 越小越平滑
        # 上一帧像素中心 (用于帧间最近邻匹配)
        self.last_center_u = float(obs['center_u'])
        self.last_center_v = float(obs['center_v'])
        # 滤波状态 (初始值 = 第一次观测)
        self.s_center_u = float(obs['center_u'])
        self.s_center_v = float(obs['center_v'])
        self.s_angle = float(obs['angle'])      # [0, 180)
        self.s_w = float(obs['w'])              # 长边 (像素)
        self.s_h = float(obs['h'])              # 短边 (像素)
        self.s_area = float(obs['area'])        # 轮廓面积 (像素²)
        # 元数据
        self.last_update_frame = frame_count
        self.miss_count = 0
        self.last_obs = obs

    def update(self, obs, frame_count, alpha=None):
        """用新观测做 EMA 更新。alpha=None 时用 self.alpha。"""
        a = alpha if alpha is not None else self.alpha
        self.last_center_u = float(obs['center_u'])
        self.last_center_v = float(obs['center_v'])
        self.s_center_u = a * float(obs['center_u']) + (1 - a) * self.s_center_u
        self.s_center_v = a * float(obs['center_v']) + (1 - a) * self.s_center_v
        self.s_angle = self._ema_angle(self.s_angle, float(obs['angle']), a)
        self.s_w = a * float(obs['w']) + (1 - a) * self.s_w
        self.s_h = a * float(obs['h']) + (1 - a) * self.s_h
        self.s_area = a * float(obs['area']) + (1 - a) * self.s_area
        # 元数据
        self.last_update_frame = frame_count
        self.miss_count = 0
        self.last_obs = obs

    @staticmethod
    def _ema_angle(old, new, alpha):
        """圆周 [0,180) 上的最短路径 EMA, 避免跨 0/180 边界跳变。"""
        diff = new - old
        while diff < -90.0:
            diff += 180.0
        while diff >= 90.0:
            diff -= 180.0
        updated = old + alpha * diff
        return updated % 180.0


# ====================================================================
# BlueBlockDetector: 蓝方块视觉识别 ROS2 节点
# ====================================================================
class BlueBlockDetector(Node):
    def __init__(self):
        super().__init__("blue_block_detector")

        # ================== HSV 颜色阈值 (ROS2 参数, 支持运行时调参) ==================
        # 蓝色 (#33CAE8) 默认阈值: H∈[95,125] S∈[80,255] V∈[50,255]
        # 调参: ros2 param set /blue_block_detector h_low 90
        # 默认值已校准 (2026-07-27): 自动曝光下 #33CAE8 蓝色有效范围
        self.declare_parameter("h_low", 90)
        self.declare_parameter("h_high", 130)
        self.declare_parameter("s_low", 50)
        self.declare_parameter("s_high", 255)
        self.declare_parameter("v_low", 40)
        self.declare_parameter("v_high", 255)

        # ================== 形态学 / 面积滤波参数 ==================
        self.declare_parameter("min_area_px", 200)        # 面积 < 此值剔除
        self.declare_parameter("roi_shrink_ratio", 0.15)  # ROI 向内收缩比例
        self.declare_parameter("min_valid_pixels", 50)    # RANSAC 最小有效像素数
        self.declare_parameter("table_ring_px", 20)       # 桌面环形区域外扩像素
        self.declare_parameter("ransac_dist_threshold", 0.005)  # RANSAC 距离阈值 (m)
        self.declare_parameter("ransac_min_inlier_ratio", 0.30) # inlier 比例下限
        self.declare_parameter("min_normal_z", 0.7)       # 法向量 c 分量下限
        self.declare_parameter("block_height_min_m", 0.008)
        self.declare_parameter("block_height_max_m", 0.055)

        # ================== 相机参数 (RealSense D455, 硬编码 fallback) ==================
        # 与 realsense_yolo_node.py 一致: camera_info 话题到达后覆盖
        self.fx = 378.394659861614
        self.fy = 379.366916262423
        self.cx = 330.140969430714
        self.cy = 246.095530649072
        self.skew = 1.25667775035477
        self.camera_matrix = np.array([
            [self.fx, self.skew, self.cx],
            [0,       self.fy,   self.cy],
            [0,       0,         1]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([
            -0.0579748391767018, 0.104170020380745,
            -0.000760277417774888, 0.000473717313218756,
            -0.0946382536306512
        ], dtype=np.float64)
        self._cam_info_received = False

        # ================== 手眼标定数据 (eye-to-hand, 相机→基座) ==================
        # 来源: /home/lxf/handeye/result/2026-07-26_16-17-46_calibration.json
        # 与 realsense_yolo_node.py 保持一致的硬编码值 (常量, 不依赖 TCP 位姿)
        self.camera_to_base_translation = np.array([
            0.6732749433043899,   # X
            0.010662122461489898, # Y
            0.597133855461562     # Z
        ])
        self.camera_to_base_quat = np.array([
            -0.6934151560947895,   # qx
            -0.7119098057466932,   # qy
            0.08845804637870601,   # qz
            0.06734258541670725    # qw
        ])
        self.camera_to_base_rpy = np.array([
            -2.9203729063228363,   # roll
            0.026795812790156897,  # pitch
            1.6000918271309976     # yaw
        ])

        # 构建 4x4 齐次变换矩阵 (相机坐标系 -> 机械臂基坐标系)
        self.rotation_matrix_cam_to_base = self.quaternion_to_rotation_matrix(
            self.camera_to_base_quat)
        self.transform_cam_to_base = np.eye(4)
        self.transform_cam_to_base[:3, :3] = self.rotation_matrix_cam_to_base
        self.transform_cam_to_base[:3, 3] = self.camera_to_base_translation
        self.transform_base_to_cam = np.linalg.inv(self.transform_cam_to_base)

        self.get_logger().info("=" * 60)
        self.get_logger().info("🔧 手眼标定参数已加载 (eye-to-hand):")
        self.get_logger().info(f"   平移: X={self.camera_to_base_translation[0]:.6f}m")
        self.get_logger().info(f"         Y={self.camera_to_base_translation[1]:.6f}m")
        self.get_logger().info(f"         Z={self.camera_to_base_translation[2]:.6f}m")
        self.get_logger().info(f"   四元数: qx={self.camera_to_base_quat[0]:.6f}")
        self.get_logger().info(f"           qy={self.camera_to_base_quat[1]:.6f}")
        self.get_logger().info(f"           qz={self.camera_to_base_quat[2]:.6f}")
        self.get_logger().info(f"           qw={self.camera_to_base_quat[3]:.6f}")
        self.get_logger().info("=" * 60)

        self.base_frame_id = self.declare_parameter("base_frame_id", "base_link").value

        # ================== 标定偏差补偿 (与 realsense_yolo_node.py 一致) ==================
        # X/Y 补偿: 修正 base 坐标偏大 (负值=减去); Z 补偿: 抬高防撞台面
        # 后续 Task 3.1 标定时再调整
        # ── eye-to-hand 标定偏移补偿 (单位: 米) ──
        # x/y_offset: 与 realsense_yolo_node.py 一致
        # z_offset_m: 蓝方块专用, 从 0.030 校正为 0.052
        #   校正依据: 桌面在基座 z≈-0.015m, 3cm 物块顶面 z≈0.015m
        #   校正前检测 surface_z=-0.007m (偏低 22mm), 增加 z_offset 22mm 补偿
        self.x_offset_m = float(self.declare_parameter('x_offset_m', -0.045).value)
        self.y_offset_m = float(self.declare_parameter('y_offset_m', -0.010).value)
        self.z_offset_m = float(self.declare_parameter('z_offset_m', 0.052).value)

        # ================== 滤波参数 (EMA + 空间关联) ==================
        self.filter_alpha = float(self.declare_parameter("filter_alpha", 0.25).value)
        self.filter_match_dist_px = float(self.declare_parameter("filter_match_dist_px", 80.0).value)
        self.filter_max_misses = int(self.declare_parameter("filter_max_misses", 10).value)
        self._tracks = {}          # {track_id: ObjectTrack}
        self._next_track_id = 0

        # ================== 初始化 ==================
        self.bridge = CvBridge()
        self.latest_rgb = None
        self.latest_rgb_stamp = None
        self.rgb_ready = False
        self.latest_depth = None           # uint16 ndarray (mm)
        self.latest_depth_stamp = 0.0
        self.frame_count = 0

        # 深度方法统计 (RANSAC vs 回退)
        self._ransac_count = 0
        self._fallback_count = 0
        self._no_depth_count = 0
        self._depth_diag_logged = False

        self.detection_count = 0
        self.status_printed = False

        # ================== 创建发布/订阅 ==================
        self.pub_detection_info = self.create_publisher(String, "/detection_info", 10)
        self.pub_annotated = self.create_publisher(
            CompressedImage, "/yolo/annotated_image/compressed", 1)

        # 订阅 (与 realsense_yolo_node.py 一致)
        self.sub_rgb = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.rgb_cb, 10)
        self.sub_depth = self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw", self.depth_cb, 10)
        self.sub_cam_info = self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self.camera_info_cb, 10)
        self.create_subscription(
            String, "/camera/exposure_ctrl", self._exposure_ctrl_cb, 10)

        # ==================== 曝光调节 (复用 realsense_yolo_node.py 模式) ====================
        # 通过 ros2 param set 调节 realsense-ros 驱动的曝光参数
        self.camera_param_node = self.declare_parameter(
            "camera_param_node", "/camera/camera").value
        self.auto_exposure_param_name = self.declare_parameter(
            "auto_exposure_param_name", "rgb_camera.enable_auto_exposure").value
        self.exposure_param_name = self.declare_parameter(
            "exposure_param_name", "rgb_camera.exposure").value
        self.exposure_min = 1
        self.exposure_max = 10000
        self.current_exposure = 100
        self.auto_exposure = True

        # ================== 定时器 ==================
        self.create_timer(0.1, self.detect_and_publish)
        self.status_timer = self.create_timer(5.0, self.print_status)
        self._depth_status_timer = self.create_timer(5.0, self._depth_status_log)

        self.get_logger().info("✅ 蓝方块检测节点启动，等待相机数据...")
        self.get_logger().info(f"🔍 HSV 阈值: H[{95},{125}] S[{80},255] V[{50},255]")
        self.get_logger().info(f"📷 眼在手外 (eye-to-hand): 相机变换为常量")
        # 默认启用自动曝光
        self.set_exposure(True)

    # ==================== 标定矩阵构建 (复用 realsense_yolo_node.py) ====================
    def quaternion_to_rotation_matrix(self, quat):
        """四元数 -> 旋转矩阵. quat: [qx, qy, qz, qw]"""
        qx, qy, qz, qw = quat
        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

    # ==================== 图像回调 ====================
    def rgb_cb(self, msg):
        """接收 RGB 图像 (cv_bridge 转 BGR8)"""
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_rgb_stamp = self._stamp_to_sec(msg.header.stamp)
            if not self.rgb_ready:
                self.rgb_ready = True
                self.get_logger().info(f"✅ RGB 就绪: {msg.width}x{msg.height}")
        except Exception as e:
            self.get_logger().error(f"RGB 转换错误: {e}")

    def depth_cb(self, msg):
        """接收 D455 对齐深度图 (uint16, mm). 像素坐标与 RGB 严格对齐."""
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
            self.latest_depth = np.ascontiguousarray(depth)
            self.latest_depth_stamp = self._stamp_to_sec(msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"深度图解析失败: {e}")
            self.latest_depth = None

    def camera_info_cb(self, msg):
        """从 /camera/camera/color/camera_info 实时更新内参 (替代硬编码 fallback).
        仅首帧更新, 覆盖 __init__ 中的硬编码 fx/fy/cx/cy。"""
        if getattr(self, '_cam_info_received', False):
            return
        try:
            k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            new_fx = float(k[0, 0]); new_fy = float(k[1, 1])
            new_cx = float(k[0, 2]); new_cy = float(k[1, 2])
            if new_fx <= 0 or new_fy <= 0:
                return
            self.fx, self.fy, self.cx, self.cy = new_fx, new_fy, new_cx, new_cy
            self.skew = float(k[0, 1])
            self.camera_matrix = k
            if len(msg.d) > 0:
                self.dist_coeffs = np.array(msg.d, dtype=np.float64)
            self._cam_info_received = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("📥 camera_info 已更新内参 (替代硬编码 fallback):")
            self.get_logger().info(f"   fx={self.fx:.4f} fy={self.fy:.4f} "
                                   f"cx={self.cx:.4f} cy={self.cy:.4f}")
            self.get_logger().info("=" * 60)
        except Exception as e:
            self.get_logger().warning(f"解析 camera_info 失败: {e}")

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    # ==================== 曝光调节 (复用 realsense_yolo_node.py) ====================
    def _exposure_ctrl_cb(self, msg):
        """接收 GUI 发来的曝光控制命令。JSON: {"auto": true} 或 {"auto": false, "value": 5000}"""
        try:
            cmd = json.loads(msg.data)
            auto = bool(cmd.get("auto", True))
            value = cmd.get("value", None)
            self.set_exposure(auto, value if value is None else int(value))
        except Exception as e:
            self.get_logger().error(f"解析曝光控制命令失败: {e} (raw: {msg.data[:100]})")

    def set_exposure(self, auto, value=None):
        """通过 ros2 param set 调节 realsense-ros 相机驱动的曝光 (异步, 不阻塞主循环)。"""
        self.auto_exposure = auto
        ros_setup = "source /opt/ros/humble/setup.bash"
        if auto:
            cmd = (f'{ros_setup} && ros2 param set {self.camera_param_node} '
                   f'{self.auto_exposure_param_name} true')
            subprocess.Popen(['bash', '-c', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("🔄 已切换为自动曝光")
        else:
            cmd = (f'{ros_setup} && ros2 param set {self.camera_param_node} '
                   f'{self.auto_exposure_param_name} false')
            subprocess.Popen(['bash', '-c', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if value is not None:
                value = max(self.exposure_min, min(self.exposure_max, int(value)))
                cmd = (f'{ros_setup} && ros2 param set {self.camera_param_node} '
                       f'{self.exposure_param_name} {value}')
                subprocess.Popen(['bash', '-c', cmd],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.current_exposure = value
                self.get_logger().info(
                    f"📷 曝光度: {int(self.current_exposure)} "
                    f"(范围: {int(self.exposure_min)} - {int(self.exposure_max)})")

    # ==================== 坐标转换 (复用 realsense_yolo_node.py) ====================
    def transform_camera_to_base(self, camera_coords):
        """相机坐标 → 基座坐标 (eye-to-hand: 常量变换矩阵 + 标定偏移补偿)"""
        if camera_coords is None:
            return None
        point_homogeneous = np.append(camera_coords, 1)
        base_coords = self.transform_cam_to_base @ point_homogeneous
        base_coords[0] += self.x_offset_m
        base_coords[1] += self.y_offset_m
        base_coords[2] += self.z_offset_m
        return base_coords[:3]

    def pixel_to_camera_coords(self, u, v, depth_m):
        """像素 (u,v) + 深度 → 相机坐标系 3D 坐标 (针孔模型反投影)"""
        x = (u - self.cx) * depth_m / self.fx
        y = (v - self.cy) * depth_m / self.fy
        z = depth_m
        return np.array([x, y, z])

    # ==================== Task 1.2: HSV 颜色分割 ====================
    def hsv_segment(self, bgr_img):
        """RGB→HSV, 蓝色掩膜 + 形态学处理 (开运算 3x3 + 闭运算 5x5)。
        HSV 阈值通过 ROS2 参数实时读取, 支持 ros2 param set 调参。"""
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        h_low = int(self.get_parameter("h_low").value)
        h_high = int(self.get_parameter("h_high").value)
        s_low = int(self.get_parameter("s_low").value)
        s_high = int(self.get_parameter("s_high").value)
        v_low = int(self.get_parameter("v_low").value)
        v_high = int(self.get_parameter("v_high").value)
        lower = np.array([h_low, s_low, v_low], dtype=np.uint8)
        upper = np.array([h_high, s_high, v_high], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        # 开运算 (3x3) 去噪
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        # 闭运算 (5x5) 填孔
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        return mask

    # ==================== Task 1.2: minAreaRect 检测 ====================
    def detect_block(self, mask):
        """从掩膜提取最大蓝色区域, 返回 minAreaRect 信息 dict 或 None。
        - findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)
        - 面积滤波 (< min_area_px 剔除)
        - 多物块选面积最大者
        - minAreaRect → 旋转角归一化到 [0°,180°), w=长边 h=短边
        """
        min_area = int(self.get_parameter("min_area_px").value)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            return None
        # 选面积最大者
        cnt = max(valid, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        rect = cv2.minAreaRect(cnt)   # ((cx, cy), (w, h), angle)
        (cx, cy), (w, h), angle = rect
        w, h, angle = self._normalize_rect(float(w), float(h), float(angle))
        return {
            'center_u': float(cx),
            'center_v': float(cy),
            'w': w,         # 长边 (像素)
            'h': h,         # 短边 (像素)
            'angle': angle, # [0, 180)
            'area': area,
            'rect': rect,   # 原始 Box2D (供 boxPoints 用)
            'contour': cnt,
        }

    @staticmethod
    def _normalize_rect(w, h, angle):
        """归一化 minAreaRect: w=长边, h=短边, angle∈[0°,180°)。
        OpenCV 4.x minAreaRect angle∈[0,90]; w<h 时长轴与之垂直, angle+=90。"""
        if w < h:
            w, h = h, w
            angle = angle + 90.0
        angle = angle % 180.0
        return w, h, angle

    @staticmethod
    def _shrink_box(box, ratio=0.15):
        """minAreaRect 四角点向中心收缩 ratio (默认 15%), 排除边缘深度跳变。"""
        center = box.mean(axis=0)
        return center + (box - center) * (1.0 - ratio)

    @staticmethod
    def _compute_pca_angle(inliers):
        """对 RANSAC inliers 的 XY 坐标做 PCA, 返回主轴方向角度 [0°,180°)。

        点云主轴方向 = 方差最大的方向, 对于长方体是长边方向。
        比 minAreaRect 旋转角更稳定, 特别是对立方体 (w≈h 时 minAreaRect 角度跳变)。

        Args:
            inliers: Nx3 数组 (相机坐标系 cam_x, cam_y, cam_z)

        Returns:
            angle_deg: 主轴方向角度 [0°,180°), 或 None (点数不足)
        """
        if len(inliers) < 10:
            return None
        # 取 XY 坐标 (相机坐标系 XY 与图像 uv 方向一致)
        xy = inliers[:, :2]  # cam_x, cam_y
        # 中心化
        xy_centered = xy - xy.mean(axis=0)
        # 协方差矩阵 (2x2)
        cov = np.cov(xy_centered.T)
        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # 最大特征值对应的特征向量 = 主轴方向
        principal_axis = eigenvectors[:, -1]  # eigh 返回升序, 最后一个是最大的
        # 主轴方向角度 (弧度), 转换到 [0°,180°)
        angle_rad = math.atan2(principal_axis[1], principal_axis[0])
        angle_deg = math.degrees(angle_rad) % 180.0
        return angle_deg

    # ==================== Task 1.3: RANSAC 平面拟合深度 ====================
    def estimate_surface_depth(self, rect, depth_img):
        """物块顶面深度估计 (相机坐标系 Z, 米)。

        流程:
          1. minAreaRect 四角点向内收缩 15% → ROI
          2. 过滤无效深度 (==0 或超出 0.15m~2.5m)
          3. 有效像素 ≥ 50 才继续
          4. 反投影为 3D 点云 (相机坐标系)
          5. Open3D segment_plane(distance_threshold=0.005) 拟合平面
          6. 校验: 法向量 c > 0.7 (朝上) 且 inlier 比例 ≥ 30%
          7. 取 inliers Z 中位数作为 surface_z_m
          8. PCA 分析 inliers XY → 点云主轴方向 (更稳定的旋转角)
          失败回退: ROI 有效深度值的 25% 分位数

        Returns: (surface_z_m, depth_method, pca_angle_deg) 或 (None, None, None)
        """
        if depth_img is None:
            return None, None, None

        shrink_ratio = float(self.get_parameter("roi_shrink_ratio").value)
        min_valid = int(self.get_parameter("min_valid_pixels").value)
        dist_thr = float(self.get_parameter("ransac_dist_threshold").value)
        min_inlier_ratio = float(self.get_parameter("ransac_min_inlier_ratio").value)
        min_normal_z = float(self.get_parameter("min_normal_z").value)

        box = cv2.boxPoints(rect)               # 4x2 float
        shrunk = self._shrink_box(box, ratio=shrink_ratio)
        xs = shrunk[:, 0]; ys = shrunk[:, 1]
        h_img, w_img = depth_img.shape
        x1 = max(0, int(math.floor(xs.min())))
        x2 = min(w_img, int(math.ceil(xs.max())) + 1)
        y1 = max(0, int(math.floor(ys.min())))
        y2 = min(h_img, int(math.ceil(ys.max())) + 1)
        if x2 <= x1 or y2 <= y1:
            return None, None

        roi = depth_img[y1:y2, x1:x2]
        # 有效深度: 非零 + 在 0.15m~2.5m (150~2500mm)
        valid_mask = (roi > 0) & (roi >= 150) & (roi <= 2500)
        ys_v, xs_v = np.nonzero(valid_mask)
        n_valid = len(xs_v)
        if n_valid == 0:
            return None, None

        depths_mm = roi[ys_v, xs_v].astype(np.float64)
        depths_m = depths_mm / 1000.0

        # 有效像素不足 50 → 直接走回退
        if n_valid < min_valid:
            return float(np.percentile(depths_m, 25)), "percentile_25", None

        # 反投影为 3D 点云 (相机坐标系)
        us = xs_v + x1
        vs = ys_v + y1
        cam_x = (us - self.cx) * depths_m / self.fx
        cam_y = (vs - self.cy) * depths_m / self.fy
        cam_z = depths_m
        pts = np.stack([cam_x, cam_y, cam_z], axis=1)

        # RANSAC 平面拟合
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            plane_model, inlier_idx = pcd.segment_plane(
                distance_threshold=dist_thr, ransac_n=3, num_iterations=2000)
            a, b, c, d = plane_model
            inliers = pts[inlier_idx]
            inlier_ratio = float(len(inliers)) / float(n_valid)
            # 校验: 法向量朝上 (c > 0.7) 且 inlier 比例 ≥ 30%
            if c > min_normal_z and inlier_ratio >= min_inlier_ratio:
                surface_z = float(np.median(inliers[:, 2]))
                # PCA 分析 inliers XY → 点云主轴方向 (更稳定的旋转角)
                pca_angle = self._compute_pca_angle(inliers)
                return surface_z, "ransac", pca_angle
            else:
                self.get_logger().debug(
                    f"RANSAC 校验失败: c={c:.3f} (需>{min_normal_z}), "
                    f"inlier_ratio={inlier_ratio:.2f} (需≥{min_inlier_ratio})",
                    throttle_duration_sec=2.0)
        except Exception as e:
            self.get_logger().warning(f"RANSAC 拟合异常: {e}", throttle_duration_sec=5.0)

        # 回退: ROI 有效深度值的 25% 分位数
        return float(np.percentile(depths_m, 25)), "percentile_25", None

    # ==================== Task 1.4: 桌面高度估算 ====================
    def estimate_table_depth(self, rect, depth_img):
        """桌面高度估算: 物块 bbox (minAreaRect 的 AABB) 外扩 ring_px 环形区域
        取深度中位数 → table_z (相机坐标系, 米)。

        相机坐标系下 Z=深度: 桌面距离相机较远 → Z 较大; 物块顶面较近 → Z 较小。
        Returns: table_z (米) 或 None
        """
        if depth_img is None:
            return None
        ring_px = int(self.get_parameter("table_ring_px").value)
        box = cv2.boxPoints(rect)
        x_min = float(box[:, 0].min()); x_max = float(box[:, 0].max())
        y_min = float(box[:, 1].min()); y_max = float(box[:, 1].max())
        h_img, w_img = depth_img.shape
        ox1 = max(0, int(math.floor(x_min)) - ring_px)
        oy1 = max(0, int(math.floor(y_min)) - ring_px)
        ox2 = min(w_img, int(math.ceil(x_max)) + ring_px)
        oy2 = min(h_img, int(math.ceil(y_max)) + ring_px)
        region = depth_img[oy1:oy2, ox1:ox2]
        # 内部 bbox 在 region 坐标系下的范围 (挖空 → 环形)
        ix1 = max(0, int(math.floor(x_min)) - ox1)
        iy1 = max(0, int(math.floor(y_min)) - oy1)
        ix2 = min(ox2 - ox1, int(math.ceil(x_max)) - ox1)
        iy2 = min(oy2 - oy1, int(math.ceil(y_max)) - oy1)
        mask = (region > 0) & (region >= 150) & (region <= 2500)
        if iy2 > iy1 and ix2 > ix1:
            mask[iy1:iy2, ix1:ix2] = False   # 挖掉物块内部 → 仅留环形
        valid = region[mask]
        if len(valid) < 5:
            return None
        return float(np.median(valid)) / 1000.0

    # ==================== 滤波: track 匹配与清理 (复用 realsense_yolo_node.py) ====================
    def _find_matching_track(self, center_u, center_v, exclude_ids=None):
        """在现有 tracks 中找像素中心距离最近的 track (贪心匹配)。
        返回 ObjectTrack 或 None。"""
        exclude_ids = exclude_ids or set()
        best_track = None
        best_dist = self.filter_match_dist_px
        for track in self._tracks.values():
            if track.track_id in exclude_ids:
                continue
            dx = track.last_center_u - center_u
            dy = track.last_center_v - center_v
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_track = track
        return best_track

    def _cleanup_stale_tracks(self, frame_count):
        """删除连续 filter_max_misses 帧未匹配的 track, 未匹配 track miss_count++。"""
        stale_ids = [tid for tid, t in self._tracks.items()
                     if t.miss_count >= self.filter_max_misses]
        for tid in stale_ids:
            del self._tracks[tid]
        for t in self._tracks.values():
            if t.last_update_frame < frame_count:
                t.miss_count += 1

    # ==================== 主检测循环 ====================
    def detect_and_publish(self):
        """主检测函数: HSV 分割 → minAreaRect → EMA → RANSAC 深度 → 坐标转换 → JSON 发布"""
        if not self.rgb_ready or self.latest_rgb is None:
            return

        self.frame_count += 1
        display_img = self.latest_rgb.copy()
        detections = []

        try:
            mask = self.hsv_segment(self.latest_rgb)
            block = self.detect_block(mask)

            if block is None:
                # 无物块: 发布空 detections (清理 track miss_count)
                self._cleanup_stale_tracks(self.frame_count)
                self._publish_annotated(display_img, detections)
                return

            self.detection_count += 1

            # ========== EMA 滤波 (匹配 track + 更新) ==========
            raw_obs = {
                'center_u': block['center_u'],
                'center_v': block['center_v'],
                'w': block['w'],
                'h': block['h'],
                'angle': block['angle'],
                'area': block['area'],
            }
            track = self._find_matching_track(block['center_u'], block['center_v'])
            if track is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                track = ObjectTrack(track_id, raw_obs, self.frame_count, self.filter_alpha)
                self._tracks[track_id] = track
            else:
                track.update(raw_obs, self.frame_count, self.filter_alpha)
            self._cleanup_stale_tracks(self.frame_count)

            # 用滤波值 (画框/标注/发布都用平滑后的值)
            center_u = track.s_center_u
            center_v = track.s_center_v
            angle = track.s_angle           # [0, 180)
            w_pix = track.s_w               # 长边 (像素)
            h_pix = track.s_h               # 短边 (像素)
            area = track.s_area

            # 滤波后的 minAreaRect 等价 Box2D (供 boxPoints/深度 ROI 用)
            rect_smoothed = ((center_u, center_v), (w_pix, h_pix), 0.0)

            # ========== Task 1.3: RANSAC 平面拟合深度 ==========
            surface_z_m, depth_method, pca_angle = self.estimate_surface_depth(
                rect_smoothed, self.latest_depth)

            # PCA 角度覆盖 minAreaRect 旋转角 (更稳定, 特别是对立方体)
            if pca_angle is not None:
                angle = pca_angle

            # ========== Task 1.4: 桌面高度 + 物块高度 ==========
            block_height_m = None
            table_z = None
            if surface_z_m is not None:
                table_z = self.estimate_table_depth(rect_smoothed, self.latest_depth)

            if surface_z_m is not None and table_z is not None:
                # 相机坐标系: 桌面远(Z大) - 顶面近(Z小) → 物块高度为正
                block_height_m = table_z - surface_z_m
                h_min = float(self.get_parameter("block_height_min_m").value)
                h_max = float(self.get_parameter("block_height_max_m").value)
                if not (h_min <= block_height_m <= h_max):
                    self.get_logger().warning(
                        f"⚠️ 物块高度 {block_height_m*1000:.1f}mm 超出预期范围 "
                        f"[{h_min*1000:.0f}, {h_max*1000:.0f}]mm "
                        f"(surface_z={surface_z_m:.3f} table_z={table_z:.3f})",
                        throttle_duration_sec=2.0)

            # ========== Task 1.4: 三维坐标计算 ==========
            base_coords = None
            block_width_m = None
            block_length_m = None
            estimated_diameter_m = None
            centroid_base_z = None
            if surface_z_m is not None:
                # 像素中心 → 相机坐标系 (用 surface_z_m 作为深度)
                cam_coords = self.pixel_to_camera_coords(center_u, center_v, surface_z_m)
                base_coords = self.transform_camera_to_base(cam_coords)
                # 物块尺寸 (针孔模型反投影, 用 surface_z_m)
                block_length_m = float(w_pix) * surface_z_m / self.fx   # 长边
                block_width_m = float(h_pix) * surface_z_m / self.fx    # 短边 (夹爪夹紧方向)
                estimated_diameter_m = block_width_m
                if base_coords is not None:
                    centroid_base_z = float(base_coords[2])  # 顶面在基座坐标系的 Z 高度

            # 深度方法统计
            if depth_method == "ransac":
                self._ransac_count += 1
            elif depth_method == "percentile_25":
                self._fallback_count += 1
            else:
                self._no_depth_count += 1

            # 首帧深度诊断
            if not self._depth_diag_logged and surface_z_m is not None:
                self._depth_diag_logged = True
                self.get_logger().info("=" * 60)
                self.get_logger().info("📐 蓝方块深度诊断 (首帧):")
                self.get_logger().info(f"   surface_z_m (相机系): {surface_z_m:.3f}m")
                self.get_logger().info(f"   table_z (相机系): {table_z:.3f}m" if table_z else "   table_z: None")
                self.get_logger().info(f"   block_height: {block_height_m*1000:.1f}mm" if block_height_m else "   block_height: None")
                self.get_logger().info(f"   depth_method: {depth_method}")
                self.get_logger().info("=" * 60)

            # ========== 轴对齐 bbox (供 JSON bbox 字段) ==========
            box_pts = cv2.boxPoints(rect_smoothed)
            x1 = float(box_pts[:, 0].min()); x2 = float(box_pts[:, 0].max())
            y1 = float(box_pts[:, 1].min()); y2 = float(box_pts[:, 1].max())

            # ========== 组装 detection (严格按 spec 格式) ==========
            det = {
                "name": "blue_block (detected)",
                "confidence": 0.95,   # HSV 颜色检测确定性高, 固定置信度
                "bbox": {
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                },
                "base_coords": {
                    "x": round(float(base_coords[0]), 4),
                    "y": round(float(base_coords[1]), 4),
                    "z": round(float(base_coords[2]), 4),
                } if base_coords is not None else None,
                "grasp_orientation": {"qx": 0, "qy": 0, "qz": 0, "qw": 1},
                "centroid_base_z": round(centroid_base_z, 4) if centroid_base_z is not None else None,
                "estimated_diameter_m": round(estimated_diameter_m, 4) if estimated_diameter_m is not None else None,
                "block_rotation_deg": round(angle, 2),
                "block_width_m": round(block_width_m, 4) if block_width_m is not None else None,
                "block_height_m": round(block_height_m, 4) if block_height_m is not None else None,
                "block_length_m": round(block_length_m, 4) if block_length_m is not None else None,
                # 注意: surface_z_m 发布的是基座坐标系下的顶面高度 (与 centroid_base_z 一致)
                # 内部变量 surface_z_m (相机系深度) 仅用于反投影计算, 不对外发布
                "surface_z_m": round(centroid_base_z, 4) if centroid_base_z is not None else None,
                "depth_method": depth_method,
            }
            detections.append(det)

            # ========== 绘制标注 (旋转矩形 + 中心点 + 角度) ==========
            self._draw_annotation(display_img, rect_smoothed, center_u, center_v,
                                  angle, block_height_m, surface_z_m, base_coords)

            # 终端日志
            if base_coords is not None:
                self.get_logger().info(
                    f"🎯 蓝方块 base=({base_coords[0]:.3f},{base_coords[1]:.3f},{base_coords[2]:.3f}) "
                    f"angle={angle:.1f}° h={block_height_m*1000:.1f}mm "
                    f"size=({block_length_m*1000:.0f}x{block_width_m*1000:.0f}x{block_height_m*1000:.0f})mm "
                    f"[{depth_method}]",
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().warning("蓝方块检测到但深度估计失败",
                                         throttle_duration_sec=2.0)

        except Exception as e:
            self.get_logger().error(f"检测错误: {e}")
            cv2.putText(display_img, f"Error: {str(e)[:50]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 发布 /detection_info (JSON) + /yolo/annotated_image/compressed
        self._publish_annotated(display_img, detections)

    # ==================== 标注绘制 ====================
    def _draw_annotation(self, img, rect, cx, cy, angle, block_height,
                         surface_z, base_coords):
        """在画面上绘制旋转矩形 + 中心点 + 角度/坐标文字。"""
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(img, [box], 0, (255, 0, 0), 2)   # 蓝色旋转矩形 (BGR)
        cv2.circle(img, (int(cx), int(cy)), 5, (0, 0, 255), -1)  # 红色中心点

        # 角度文字 (中心点旁)
        cv2.putText(img, f"angle={angle:.1f}",
                    (int(cx) + 8, int(cy) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

        # 顶部汇总信息
        info_lines = [f"Blue Block  angle={angle:.1f}"]
        if surface_z is not None:
            info_lines.append(f"surface_z={surface_z*1000:.0f}mm")
        if block_height is not None:
            info_lines.append(f"height={block_height*1000:.0f}mm")
        if base_coords is not None:
            info_lines.append(
                f"base=({base_coords[0]:.2f},{base_coords[1]:.2f},{base_coords[2]:.2f})")
        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # ==================== 发布 ====================
    def _publish_annotated(self, display_img, detections):
        """发布 /detection_info (JSON) + /yolo/annotated_image/compressed (JPEG)。"""
        # /detection_info (严格按 spec 格式: {"detections": [...]})
        info_dict = {"detections": detections}
        info_msg = String()
        info_msg.data = json.dumps(info_dict, ensure_ascii=False)
        self.pub_detection_info.publish(info_msg)

        # /yolo/annotated_image/compressed (JPEG)
        try:
            ok, jpeg_buf = cv2.imencode('.jpg', display_img,
                                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                cimg = CompressedImage()
                cimg.header.stamp = self.get_clock().now().to_msg()
                cimg.format = "jpeg"
                cimg.data = jpeg_buf.tobytes()
                self.pub_annotated.publish(cimg)
        except Exception:
            pass

    # ==================== 状态日志 ====================
    def print_status(self):
        """首次状态打印"""
        if not self.status_printed:
            self.status_printed = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("📊 状态统计:")
            self.get_logger().info(f"  - RGB 数据: {'就绪' if self.rgb_ready else '等待中'}")
            self.get_logger().info(f"  - 深度数据: {'就绪' if self.latest_depth is not None else '等待中'}")
            self.get_logger().info(f"  - 已处理帧数: {self.frame_count}")
            self.get_logger().info(f"  - 检测到蓝方块: {self.detection_count} 次")
            self.get_logger().info("=" * 60)
            self.status_timer.cancel()

    def _depth_status_log(self):
        """周期诊断 (每5s): 深度方法统计 (RANSAC vs 回退 vs 无深度)"""
        total = self._ransac_count + self._fallback_count + self._no_depth_count
        if total > 0:
            self.get_logger().info(
                f"📏 深度方法统计: RANSAC={self._ransac_count} "
                f"percentile_25={self._fallback_count} 无深度={self._no_depth_count} "
                f"(总计 {total})")
        elif self.latest_depth is None:
            self.get_logger().error(
                "❌ D455 深度图未收到 (latest_depth=None)! "
                "话题 /camera/camera/aligned_depth_to_color/image_raw 无数据。"
                "修复: 相机以 align_depth.enable:=true 启动。")
        # 重置周期计数
        self._ransac_count = 0
        self._fallback_count = 0
        self._no_depth_count = 0


def main(args=None):
    rclpy.init(args=args)
    node = BlueBlockDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
