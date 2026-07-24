#!/usr/bin/env python3
"""grasp_pose_node.py

GraspNet-1Billion 抓取位姿生成节点 (eye-to-hand 配置).

订阅:
  - /camera/camera/aligned_depth_to_color/image_raw  (D455 对齐深度图, uint16 mm)
  - /camera/camera/color/image_raw                   (RGB)
  - /camera/camera/color/camera_info                 (内参)

服务:
  - /generate_grasp_pose  (agx_arm_msgs/srv/GenerateGraspPose)

发布:
  - /grasp_pose_marker   (visualization_msgs/MarkerArray, RViz 可视化)

设计要点:
  - sys.path 注入 orange_dataset/.venv (含 torch/open3d/graspnet-baseline)
  - 时间戳缓冲: dict + stamp_ns 键, 二分查找容差 100ms (depth)
  - 点云生成: GraspNet 自带 create_point_cloud_from_depth_image
  - bbox 裁剪: 30% margin 膨胀
  - 预处理: voxel_down_sample(0.005) + statistical_outlier_removal
  - z-down 过滤: approach_dir 与 [0,0,-1] 夹角 < 30°
  - 坐标变换链: camera_color_optical_frame -> base_link (eye-to-hand 常量变换, 不需要 TCP 位姿)
"""

import os
import sys
import json
import time
import math
import threading
from collections import deque
from bisect import bisect_left

import numpy as np

# ==================== sys.path 注入 ====================
# GraspNet-1Billion 仓库 (提供 models/, utils/, pointnet2/)
_GRASPNET_ROOT = "/home/lxf/graspnet/graspnet-baseline"
if _GRASPNET_ROOT not in sys.path:
    sys.path.insert(0, _GRASPNET_ROOT)
sys.path.insert(0, os.path.join(_GRASPNET_ROOT, "models"))
sys.path.insert(0, os.path.join(_GRASPNET_ROOT, "utils"))
sys.path.insert(0, os.path.join(_GRASPNET_ROOT, "pointnet2"))
sys.path.insert(0, os.path.join(_GRASPNET_ROOT, "knn"))

# 工作区根 (提供 agx_arm_msgs)
_WS_ROOT = "/home/lxf/agx_arm_ws"
if _WS_ROOT not in sys.path:
    sys.path.insert(0, _WS_ROOT)

# ==================== ROS2 imports ====================
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Header

# agx_arm_msgs (colcon 编译后 install 路径)
# 注: 实际安装路径是 local/lib/python3.10/dist-packages (ROS2 Humble 默认)
# 同时兼容 lib/python3.10/site-packages (自定义配置)
_AGX_MSGS_INSTALL_PATHS = [
    "/home/lxf/agx_arm_ws/install/agx_arm_msgs/local/lib/python3.10/dist-packages",
    "/home/lxf/agx_arm_ws/install/agx_arm_msgs/lib/python3.10/site-packages",
]
for _p in _AGX_MSGS_INSTALL_PATHS:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from agx_arm_msgs.srv import GenerateGraspPose
from agx_arm_msgs.msg import GraspCandidate


# ==================== torch / open3d / graspnet imports ====================
import torch
import open3d as o3d
from graspnetAPI import GraspGroup

from graspnet import GraspNet, pred_decode
from data_utils import CameraInfo as GraspNetCameraInfo, create_point_cloud_from_depth_image


# ==================== 手眼标定数据 (eye-to-hand 常量变换) ====================
# 来源: /home/lxf/handeye/result/2026-07-25_03-57-46_calibration.json
# 与 realsense_yolo_node.py 中完全一致
_HANDEYE_JSON_PATH = "/home/lxf/handeye/result/2026-07-25_03-57-46_calibration.json"


def _quat_to_rotmat(qx, qy, qz, qw):
    """四元数 -> 3x3 旋转矩阵 (与 tf_transformations 一致)"""
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float64)


def _load_handeye_matrix():
    """加载手眼标定矩阵 (相机 -> 基座, eye-to-hand 常量变换)

    JSON 格式 (与 handeye_calibration 节点输出一致):
        {
            "position": [x, y, z],            # 平移 (米)
            "orientation": [qx, qy, qz, qw],  # 四元数
            "rpy": [[roll, pitch, yaw]]       # RPY (弧度, 备用)
        }

    Returns:
        4x4 numpy array, camera_color_optical_frame -> base_link (eye-to-hand, 常量变换)
    """
    try:
        with open(_HANDEYE_JSON_PATH, "r") as f:
            data = json.load(f)
        # 解析平移: position 是 [x, y, z] 数组
        pos = data.get("position", [0.0, 0.0, 0.0])
        tx = float(pos[0]); ty = float(pos[1]); tz = float(pos[2])
        # 解析四元数: orientation 是 [qx, qy, qz, qw] 数组
        ori = data.get("orientation", [0.0, 0.0, 0.0, 1.0])
        qx = float(ori[0]); qy = float(ori[1]); qz = float(ori[2]); qw = float(ori[3])
    except Exception as e:
        raise RuntimeError(f"无法加载手眼标定数据 {_HANDEYE_JSON_PATH}: {e}")

    R = _quat_to_rotmat(qx, qy, qz, qw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


class GraspPoseNode(Node):
    """GraspNet 抓取位姿生成节点 (eye-to-hand 配置).

    主要职责:
        1. 维护深度图的时间戳缓冲 (用于按 header.stamp 查找历史深度图)
        2. 收到 service 请求时, 按 header.stamp 查找对应时刻的深度图
        3. 按 bbox 裁剪点云, 调用 GraspNet 推理
        4. 过滤 z-down 候选, 用常量变换矩阵 T_cam_to_base 变换到 base_link
        5. 返回 top-N 候选 + 发布 RViz 可视化

    注: eye-to-hand 配置下相机外参为常量, 不需要订阅 /feedback/tcp_pose.
    """

    def __init__(self):
        super().__init__("grasp_pose_node")

        # ==================== 参数声明 ====================
        # GraspNet 模型
        self.checkpoint_path = self.declare_parameter(
            "graspnet_checkpoint",
            "/home/lxf/graspnet/checkpoints/checkpoint-rs.tar"
        ).value
        self.num_point = int(self.declare_parameter("num_point", 20000).value)
        self.num_view = int(self.declare_parameter("num_view", 300).value)
        # 夹爪开合宽度 (m). GraspNet cylinder_radius = max_width / 2
        self.max_gripper_width = float(self.declare_parameter("max_gripper_width", 0.10).value)
        # 抓取过滤
        self.score_threshold = float(self.declare_parameter("score_threshold", 0.5).value)
        self.approach_angle_max_deg = float(self.declare_parameter("approach_angle_max_deg", 30.0).value)
        self.max_candidates = int(self.declare_parameter("max_candidates", 5).value)
        # 工作空间过滤 (base_link 系下, m)
        self.workspace_min = self.declare_parameter(
            "workspace_min", "[-0.5, -0.5, 0.0]"
        ).value
        self.workspace_max = self.declare_parameter(
            "workspace_max", "[0.8, 0.5, 0.6]"
        ).value
        # 点云预处理
        self.voxel_size = float(self.declare_parameter("voxel_size", 0.005).value)
        self.bbox_margin_ratio = float(self.declare_parameter("bbox_margin_ratio", 0.3).value)
        # D455 近距离深度偏差校正 (米): 从原始深度中减去此值, 修正 <0.4m 近距离偏大.
        # eye-to-hand 配置下相机距离物体较远, 不需要近距离校正, 设为 0.0.
        # 与 realsense_yolo_node.py 的 depth_offset_m 保持一致.
        self.depth_offset_m = float(self.declare_parameter("depth_offset_m", 0.0).value)
        # 短轴对齐: 用 PCA 计算物体短轴, 让夹爪 X 轴(开合方向)对齐短轴, 夹住较短方向
        self.align_gripper_to_short_axis = bool(
            self.declare_parameter("align_gripper_to_short_axis", True).value
        )
        # 长宽比阈值: 长轴特征值/短轴特征值 >= 此值才视为细长物体并触发重写
        self.elongation_ratio_threshold = float(
            self.declare_parameter("elongation_ratio_threshold", 1.5).value
        )
        # 时间同步 (仅深度; eye-to-hand 不需要 TCP 位姿同步)
        # depth_sync_tolerance_s 保留可调, 用于 _lookup_depth_by_stamp
        self.depth_sync_tolerance_s = float(self.declare_parameter("depth_sync_tolerance_s", 0.10).value)
        self.buffer_max_age_s = float(self.declare_parameter("buffer_max_age_s", 30.0).value)
        # 碰撞检测
        self.collision_thresh = float(self.declare_parameter("collision_thresh", 0.01).value)
        # 话题/坐标系
        self.depth_topic = self.declare_parameter(
            "depth_topic", "/camera/camera/aligned_depth_to_color/image_raw"
        ).value
        self.rgb_topic = self.declare_parameter(
            "rgb_topic", "/camera/camera/color/image_raw"
        ).value
        self.camera_info_topic = self.declare_parameter(
            "camera_info_topic", "/camera/camera/color/camera_info"
        ).value
        self.base_frame_id = self.declare_parameter("base_frame_id", "base_link").value

        # 解析 workspace 参数 (字符串 -> list)
        try:
            if isinstance(self.workspace_min, str):
                self.workspace_min = json.loads(self.workspace_min)
            if isinstance(self.workspace_max, str):
                self.workspace_max = json.loads(self.workspace_max)
            self.workspace_min = [float(v) for v in self.workspace_min]
            self.workspace_max = [float(v) for v in self.workspace_max]
        except Exception as e:
            self.get_logger().warn(f"解析 workspace 参数失败 ({e}), 使用默认值")
            self.workspace_min = [-0.5, -0.5, 0.0]
            self.workspace_max = [0.8, 0.5, 0.6]

        # ==================== 手眼标定矩阵 (eye-to-hand 常量变换) ====================
        try:
            self.T_cam_to_base = _load_handeye_matrix()
            self.get_logger().info("=" * 60)
            self.get_logger().info("🔧 手眼标定矩阵已加载 (cam -> base_link, eye-to-hand):")
            self.get_logger().info(f"   平移: {self.T_cam_to_base[:3, 3]}")
            self.get_logger().info("=" * 60)
        except Exception as e:
            self.get_logger().error(f"❌ 手眼标定加载失败: {e}")
            self.T_cam_to_base = np.eye(4)

        # ==================== 时间戳缓冲 ====================
        # 用 deque 存 (stamp_ns, data), 按 stamp_ns 升序
        # 深度缓冲: 30fps × 30s = 900 帧 (覆盖用户点击延迟)
        # eye-to-hand 不需要 TCP 位姿缓冲
        self._depth_buffer = deque(maxlen=900)        # [(stamp_ns, depth_uint16)]
        self._buffer_lock = threading.Lock()
        self._last_cleanup_time = time.time()

        # ==================== 相机内参 ====================
        self._camera_info = None  # sensor_msgs/CameraInfo

        # ==================== 回调组 ====================
        # 服务回调用独占组, 避免与订阅回调竞争 (推理耗时长)
        self._cb_sub = MutuallyExclusiveCallbackGroup()
        self._cb_service = MutuallyExclusiveCallbackGroup()

        # ==================== 订阅者 ====================
        self._depth_sub = self.create_subscription(
            Image, self.depth_topic, self._depth_cb, 10,
            callback_group=self._cb_sub
        )
        self._rgb_sub = self.create_subscription(
            Image, self.rgb_topic, self._rgb_cb, 10,
            callback_group=self._cb_sub
        )
        self._cam_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self._cam_info_cb, 10,
            callback_group=self._cb_sub
        )
        # 注: eye-to-hand 配置不需要订阅 /feedback/tcp_pose
        # 相机 -> base_link 的变换是常量 (self.T_cam_to_base), 与 TCP 位姿无关

        # ==================== 服务与发布 ====================
        self._grasp_service = self.create_service(
            GenerateGraspPose, "/generate_grasp_pose",
            self._handle_grasp_request,
            callback_group=self._cb_service
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, "/grasp_pose_marker", 10
        )

        # ==================== 加载 GraspNet 模型 ====================
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._net = None
        self._model_loaded = False
        self._load_model()

        # ==================== 清理定时器 ====================
        self._cleanup_timer = self.create_timer(5.0, self._cleanup_buffers)

        self.get_logger().info("✅ grasp_pose_node 启动完成, 等待 service 请求...")

    # ========================================================================
    # 模型加载
    # ========================================================================
    def _load_model(self):
        """加载 GraspNet-1Billion checkpoint"""
        try:
            if not os.path.exists(self.checkpoint_path):
                self.get_logger().error(
                    f"❌ GraspNet checkpoint 不存在: {self.checkpoint_path}"
                )
                self.get_logger().error(
                    "   请从 https://drive.google.com/drive/folders/1xj2QJZqW3FvNnP8rDhxLRuGziB4cbDLp "
                    "下载 checkpoint-rs.tar (RealSense) 到 /home/lxf/graspnet/checkpoints/"
                )
                return

            # cylinder_radius = max_width / 2 = 0.05 (夹爪开合 10cm)
            cylinder_radius = self.max_gripper_width / 2.0
            net = GraspNet(
                input_feature_dim=0,
                num_view=self.num_view,
                num_angle=12,
                num_depth=4,
                cylinder_radius=cylinder_radius,
                hmin=-0.02,
                hmax_list=[0.01, 0.02, 0.03, 0.04],
                is_training=False,
            )
            net.to(self._device)
            checkpoint = torch.load(self.checkpoint_path, map_location=self._device)
            net.load_state_dict(checkpoint["model_state_dict"])
            net.eval()
            self._net = net
            self._model_loaded = True
            self.get_logger().info(
                f"✅ GraspNet 模型加载成功 (epoch={checkpoint.get('epoch', '?')}, "
                f"device={self._device}, cylinder_radius={cylinder_radius})"
            )
        except Exception as e:
            import traceback
            self.get_logger().error(f"❌ GraspNet 模型加载失败: {e}\n{traceback.format_exc()}")

    # ========================================================================
    # 订阅回调
    # ========================================================================
    def _depth_cb(self, msg: Image):
        """缓存深度图 (uint16, mm)"""
        try:
            # 假设编码是 16UC1 (D455 默认)
            if msg.encoding not in ("16UC1", "mono16"):
                self.get_logger().warn_once(
                    f"深度图编码 {msg.encoding} 非 16UC1, 可能解析异常",
                    throttle_duration_sec=10.0
                )
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            ).copy()
            stamp_ns = self._stamp_to_ns(msg.header.stamp)
            with self._buffer_lock:
                self._depth_buffer.append((stamp_ns, depth))
        except Exception as e:
            self.get_logger().warn(f"深度回调异常: {e}")

    def _rgb_cb(self, msg: Image):
        """缓存 RGB 图像 (供点云着色)"""
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.encoding in ("rgb8", "RGB8"):
                self._latest_rgb = arr.reshape(msg.height, msg.width, 3)
            elif msg.encoding in ("bgr8", "BGR8"):
                self._latest_rgb = arr.reshape(msg.height, msg.width, 3)[:, :, ::-1]
            else:
                self._latest_rgb = None
            self._latest_rgb_stamp = self._stamp_to_ns(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"RGB 回调异常: {e}")

    def _cam_info_cb(self, msg: CameraInfo):
        """缓存相机内参"""
        if self._camera_info is None:
            self._camera_info = msg
            self.get_logger().info(
                f"📷 相机内参已加载: fx={msg.k[0]:.2f}, fy={msg.k[4]:.2f}, "
                f"cx={msg.k[2]:.2f}, cy={msg.k[5]:.2f}, {msg.width}x{msg.height}"
            )

    # ========================================================================
    # 时间戳缓冲查找 (二分查找)
    # ========================================================================
    @staticmethod
    def _stamp_to_ns(stamp) -> int:
        """builtin_interfaces/Time -> 纳秒整数"""
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    @staticmethod
    def _stamp_float_to_ns(stamp_float: float) -> int:
        """浮点秒 -> 纳秒整数"""
        return int(stamp_float * 1_000_000_000)

    def _lookup_depth_by_stamp(self, target_ns: int):
        """二分查找最接近 target_ns 的深度图

        Args:
            target_ns: 目标时间戳 (纳秒)
        Returns:
            (stamp_ns, depth_array) 或 None (超出容差)
        """
        tol_ns = int(self.depth_sync_tolerance_s * 1_000_000_000)
        with self._buffer_lock:
            if not self._depth_buffer:
                return None
            # 提取时间戳数组
            stamps = [s for s, _ in self._depth_buffer]
            idx = bisect_left(stamps, target_ns)
            candidates = []
            if idx < len(stamps):
                candidates.append(idx)
            if idx > 0:
                candidates.append(idx - 1)
            best = None
            best_diff = None
            for i in candidates:
                diff = abs(stamps[i] - target_ns)
                if best_diff is None or diff < best_diff:
                    best_diff = diff
                    best = i
            if best is None or best_diff > tol_ns:
                return None
            return self._depth_buffer[best]

    def _cleanup_buffers(self):
        """定时清理过期缓冲 (> buffer_max_age_s)"""
        now_ns = self._stamp_float_to_ns(time.time())
        cutoff = now_ns - int(self.buffer_max_age_s * 1_000_000_000)
        with self._buffer_lock:
            while self._depth_buffer and self._depth_buffer[0][0] < cutoff:
                self._depth_buffer.popleft()

    # ========================================================================
    # 点云生成与裁剪
    # ========================================================================
    def _depth_to_pointcloud(self, depth_uint16, camera_info: CameraInfo):
        """用 GraspNet 自带工具将深度图转为点云 (相机系)

        Args:
            depth_uint16: (H, W) uint16 深度图, 单位 mm
            camera_info: ROS CameraInfo
        Returns:
            open3d.geometry.PointCloud (带颜色, 若有 RGB 缓存)
        """
        # CameraInfo.k = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx = camera_info.k[0]
        fy = camera_info.k[4]
        cx = camera_info.k[2]
        cy = camera_info.k[5]
        height = camera_info.height
        width = camera_info.width
        # GraspNet CameraInfo: scale = 1000 (mm -> m)
        cam = GraspNetCameraInfo(width, height, fx, fy, cx, cy, 1000.0)
        # ── D455 近距离偏差校正: 从深度图中减去 offset (mm) ──
        # 修正 D455 在 <0.4m 近距离的系统性偏大, 使点云与真实物体位置匹配
        if self.depth_offset_m != 0.0:
            correction_mm = int(round(self.depth_offset_m * 1000))
            depth_uint16 = np.clip(
                depth_uint16.astype(np.int32) - correction_mm, 1, 65535
            ).astype(np.uint16)
        depth_m = depth_uint16.astype(np.float32)  # GraspNet 内部除以 scale
        cloud = create_point_cloud_from_depth_image(depth_m, cam, organized=True)
        # cloud shape: (H, W, 3), 点在相机系

        # 构造 Open3D 点云 (只保留 depth>0 的点)
        mask = depth_uint16 > 0
        points = cloud[mask]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # 着色 (若有 RGB)
        rgb = getattr(self, "_latest_rgb", None)
        if rgb is not None and rgb.shape[:2] == (height, width):
            colors = rgb[mask].astype(np.float64) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd, cloud, mask

    def _crop_by_bbox(self, pcd, cloud_organized, mask, bbox, margin_ratio):
        """按 bbox 裁剪点云 (带 margin 膨胀)

        Args:
            pcd: Open3D PointCloud (已 mask)
            cloud_organized: (H, W, 3) organized 点云
            mask: (H, W) bool, depth>0
            bbox: [x1, y1, x2, y2] 像素坐标
            margin_ratio: 膨胀比例 (0.3 = 30%)
        Returns:
            Open3D PointCloud (裁剪后)
        """
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        # 膨胀
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        dx = int(w * margin_ratio)
        dy = int(h * margin_ratio)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(cloud_organized.shape[1] - 1, x2 + dx)
        y2 = min(cloud_organized.shape[0] - 1, y2 + dy)

        # 取 bbox 区域内的有效点
        sub_mask = np.zeros_like(mask)
        sub_mask[y1:y2 + 1, x1:x2 + 1] = True
        sub_mask = sub_mask & mask
        points = cloud_organized[sub_mask]
        pcd_crop = o3d.geometry.PointCloud()
        pcd_crop.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        # 着色
        rgb = getattr(self, "_latest_rgb", None)
        if rgb is not None and rgb.shape[:2] == cloud_organized.shape[:2]:
            colors = rgb[sub_mask].astype(np.float64) / 255.0
            pcd_crop.colors = o3d.utility.Vector3dVector(colors)
        return pcd_crop

    def _preprocess(self, pcd):
        """点云预处理: voxel downsample + SOR"""
        if len(pcd.points) == 0:
            return pcd
        pcd = pcd.voxel_down_sample(self.voxel_size)
        if len(pcd.points) > 20:
            pcd = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)[0]
        return pcd

    # ========================================================================
    # GraspNet 推理
    # ========================================================================
    def _run_graspnet(self, pcd):
        """调用 GraspNet 推理

        Args:
            pcd: Open3D PointCloud (相机系)
        Returns:
            GraspGroup 或空 list
        """
        if not self._model_loaded:
            self.get_logger().error("GraspNet 模型未加载, 无法推理")
            return []

        try:
            points = np.asarray(pcd.points, dtype=np.float32)
            colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.zeros_like(points)
            if len(points) < 100:
                self.get_logger().warn(f"点云点数过少 ({len(points)}), 跳过推理")
                return []

            # 采样到 num_point (不足则重复采样)
            if len(points) >= self.num_point:
                idxs = np.random.choice(len(points), self.num_point, replace=False)
            else:
                idxs1 = np.arange(len(points))
                idxs2 = np.random.choice(len(points), self.num_point - len(points), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = points[idxs]
            color_sampled = colors[idxs]

            # 转为 tensor
            cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(self._device)
            end_points = {
                "point_clouds": cloud_tensor,
                "cloud_colors": color_sampled,
            }
            with torch.no_grad():
                end_points = self._net(end_points)
                grasp_preds = pred_decode(end_points)

            gg_array = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(gg_array)
            return gg
        except Exception as e:
            import traceback
            self.get_logger().error(f"GraspNet 推理失败: {e}\n{traceback.format_exc()}")
            return []

    # ========================================================================
    # 抓取过滤与排序
    # ========================================================================
    def _filter_grasps(self, gg: GraspGroup):
        """z-down 过滤 + score 过滤

        Args:
            gg: GraspGroup
        Returns:
            list of dict: [{rotation, translation, score, width, depth, approach_angle_deg}, ...]
            按 score 降序
        """
        if len(gg) == 0:
            return []

        # 抓取坐标系约定 (GraspNet):
        # rotation_matrix[:, 2] = approach direction (指向被夹物体)
        # 我们要求 approach 方向接近 -z (从上往下夹)
        # 计算与 [0, 0, -1] 的夹角
        cos_threshold = math.cos(math.radians(self.approach_angle_max_deg))
        results = []
        for i in range(len(gg)):
            grasp = gg[i]
            R = grasp.rotation_matrix  # (3, 3) numpy
            approach = R[:, 2]  # approach direction (相机系)
            # 转到 base 系后再判定 z-down (因为相机可能斜着看)
            # 简化: 直接在 base 系下计算 (后续 _transform_grasp_to_base 会处理)
            # 这里先用相机系粗过滤 (相机大致朝下, 误差可接受)
            # 实际过滤在 transform 后做
            score = float(grasp.score)
            if score < self.score_threshold:
                continue
            results.append({
                "rotation": R,
                "translation": grasp.translation,
                "score": score,
                "width": float(grasp.width),
                "depth": float(grasp.depth),
                "approach_cam": approach,
            })
        # 按 score 降序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ========================================================================
    # 坐标变换
    # ========================================================================
    def _transform_grasp_to_base(self, grasp):
        """把相机系下的抓取位姿变换到 base_link 系 (eye-to-hand 常量变换)

        Args:
            grasp: dict with "rotation" (3x3), "translation" (3,)
        Returns:
            dict with "position" (3,), "rotation_matrix" (3,3), "approach" (3,), "approach_angle_deg" (float)
            或 None (workspace 过滤失败)
        """
        # T_cam (相机系下的抓取位姿)
        T_cam = np.eye(4, dtype=np.float64)
        T_cam[:3, :3] = grasp["rotation"]
        T_cam[:3, 3] = grasp["translation"]

        # eye-to-hand: 相机外参为常量, T_base = T_cam_to_base @ T_cam
        # 变换链: 相机系 -> base系 (常量变换, 不依赖 TCP 位姿)
        T_base = self.T_cam_to_base @ T_cam

        pos = T_base[:3, 3]
        R_base = T_base[:3, :3]
        approach_base = R_base[:, 2]  # approach direction in base frame

        # ── 诊断日志 (仅对 best 候选打印, 避免刷屏) ──
        if not getattr(self, "_diag_logged", False):
            self._diag_logged = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("🔍 坐标变换诊断 (首个候选, eye-to-hand):")
            self.get_logger().info(f"   [1] GraspNet 输出 (cam 系): "
                                   f"translation={grasp['translation']}, score={grasp.get('score', '?')}")
            self.get_logger().info(f"   [2] 手眼标定 T_cam_to_base 平移: {self.T_cam_to_base[:3, 3]}")
            self.get_logger().info(f"   [3] base系下抓取点 pos: {pos}")
            self.get_logger().info(f"   [4] approach (base 系): {approach_base}")
            # 反推: 相机看到物体的深度 (z 分量)
            cam_z = float(grasp['translation'][2])
            self.get_logger().info(f"   [5] 物体到相机深度 (cam z): {cam_z:.3f}m")
            self.get_logger().info("=" * 60)

        # z-down 过滤: approach 与 [0, 0, -1] 的夹角
        cos_angle = float(np.dot(approach_base, np.array([0.0, 0.0, -1.0])))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg >= self.approach_angle_max_deg:
            return None

        # workspace 过滤
        ws_min = self.workspace_min
        ws_max = self.workspace_max
        if not (ws_min[0] <= pos[0] <= ws_max[0] and
                ws_min[1] <= pos[1] <= ws_max[1] and
                ws_min[2] <= pos[2] <= ws_max[2]):
            return None

        return {
            "position": pos,
            "rotation_matrix": R_base,
            "approach": approach_base,
            "approach_angle_deg": angle_deg,
        }

    # ========================================================================
    # 短轴对齐 (PCA)
    # ========================================================================
    def _compute_short_axis_base(self, pcd_crop):
        """对裁剪点云做 PCA 计算物体短轴方向 (base 系)

        用于把夹爪开合方向 (X 轴) 对齐物体短轴, 使夹爪夹住较短方向.

        Args:
            pcd_crop: Open3D PointCloud (相机系)
        Returns:
            (short_axis_3d, long_axis_3d, elongation_ratio) 或 (None, None, 0.0)
            short/long_axis_3d: base 系 XY 平面内的单位向量 (z=0)
            elongation_ratio: 长轴特征值 / 短轴特征值 (< threshold 表示近圆形)
        """
        try:
            pts = np.asarray(pcd_crop.points, dtype=np.float64)
            if len(pts) < 30:
                return None, None, 0.0
            # eye-to-hand: 相机 -> base 系常量变换 (与 _transform_grasp_to_base 同链)
            T_base = self.T_cam_to_base
            pts_h = np.hstack([pts, np.ones((len(pts), 1))])
            pts_base = (T_base @ pts_h.T).T[:, :3]

            # 高度带通滤波: 剔除传送带平面, 只留物体 (±3cm 围绕中位 z)
            median_z = float(np.median(pts_base[:, 2]))
            z_mask = np.abs(pts_base[:, 2] - median_z) < 0.03
            pts_obj = pts_base[z_mask]
            if len(pts_obj) < 30:
                pts_obj = pts_base  # 滤波后过少则用全部

            # 2D PCA (XY 平面)
            pts_xy = pts_obj[:, :2]
            pts_xy_centered = pts_xy - pts_xy.mean(axis=0)
            cov = np.cov(pts_xy_centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)  # 升序: eigvals[0] < eigvals[1]
            long_xy = eigvecs[:, 1]   # 最大特征值方向 = 长轴
            short_xy = eigvecs[:, 0]  # 最小特征值方向 = 短轴
            short_axis_3d = np.array([short_xy[0], short_xy[1], 0.0])
            short_axis_3d = short_axis_3d / max(np.linalg.norm(short_axis_3d), 1e-9)
            long_axis_3d = np.array([long_xy[0], long_xy[1], 0.0])
            long_axis_3d = long_axis_3d / max(np.linalg.norm(long_axis_3d), 1e-9)
            min_ev = max(float(eigvals[0]), 1e-9)
            elongation_ratio = float(eigvals[1]) / min_ev
            return short_axis_3d, long_axis_3d, elongation_ratio
        except Exception as e:
            self.get_logger().warn(f"PCA 短轴计算失败: {e}")
            return None, None, 0.0

    # ========================================================================
    # service 回调
    # ========================================================================
    def _handle_grasp_request(self, request, response):
        """处理 GenerateGraspPose service 请求

        流程:
            1. 时间同步查找深度图 (eye-to-hand: 不需要 TCP 位姿)
            2. 生成点云, bbox 裁剪, 预处理
            3. GraspNet 推理
            4. z-down + score 过滤, 按 score 降序
            5. 坐标变换到 base_link + workspace 过滤
            6. 构造 GraspCandidate[] Response, 发布 RViz Marker
        """
        t_start = time.time()
        response.success = False
        response.error_msg = ""
        response.candidates = []
        response.best_score = 0.0

        # 检查模型
        if not self._model_loaded:
            response.error_msg = "graspnet_model_not_loaded"
            self.get_logger().error(f"❌ service 调用失败: {response.error_msg}")
            return response

        # 检查相机内参
        if self._camera_info is None:
            response.error_msg = "camera_info_not_received"
            self.get_logger().error(f"❌ service 调用失败: {response.error_msg}")
            return response

        # 提取请求参数
        # request.header.stamp: 检测时刻 (用于时间同步)
        # request.bbox: sensor_msgs/RegionOfInterest (x_offset, y_offset, width, height)
        # request.class_name: 物体类别 (用于日志)
        # request.max_candidates: top-N
        max_n = int(request.max_candidates) if request.max_candidates > 0 else self.max_candidates

        target_ns = self._stamp_to_ns(request.header.stamp)
        if target_ns == 0:
            # 用最新深度
            with self._buffer_lock:
                if self._depth_buffer:
                    target_ns = self._depth_buffer[-1][0]
                else:
                    response.error_msg = "depth_buffer_empty"
                    return response

        bbox_roi = request.bbox
        bbox = [
            float(bbox_roi.x_offset),
            float(bbox_roi.y_offset),
            float(bbox_roi.x_offset + bbox_roi.width),
            float(bbox_roi.y_offset + bbox_roi.height),
        ]
        if bbox_roi.width == 0 or bbox_roi.height == 0:
            response.error_msg = "invalid_bbox"
            return response

        self.get_logger().info(
            f"📥 收到抓取请求: class={request.class_name}, bbox={bbox}, "
            f"stamp_ns={target_ns}, max_candidates={max_n}"
        )
        # 重置诊断标志 (每次请求打印一次坐标变换诊断)
        self._diag_logged = False

        # 1. 时间同步查找
        # 若按 stamp 查找失败 (detection_stamp 过时), 回退到最新数据 + 警告
        # 这对静态物体 (传送带停止后) 是可接受的
        depth_entry = self._lookup_depth_by_stamp(target_ns)
        if depth_entry is None:
            with self._buffer_lock:
                if self._depth_buffer:
                    depth_entry = self._depth_buffer[-1]
                    self.get_logger().warn(
                        f"⚠️ 深度同步失败 (target_ns={target_ns}, "
                        f"buffer={len(self._depth_buffer)}), 回退到最新深度图"
                    )
                else:
                    response.error_msg = "depth_sync_timeout"
                    self.get_logger().warn(
                        f"❌ 深度同步失败且缓冲为空: target_ns={target_ns}"
                    )
                    return response
        depth_stamp_ns, depth_img = depth_entry

        # 注: eye-to-hand 配置不需要查找 TCP 位姿
        # 相机 -> base_link 的变换为常量 (self.T_cam_to_base), 在 _transform_grasp_to_base 中使用

        # 2. 生成点云 + bbox 裁剪
        try:
            pcd_full, cloud_organized, mask = self._depth_to_pointcloud(
                depth_img, self._camera_info
            )
            pcd_crop = self._crop_by_bbox(
                pcd_full, cloud_organized, mask, bbox, self.bbox_margin_ratio
            )
            pcd_crop = self._preprocess(pcd_crop)
        except Exception as e:
            import traceback
            response.error_msg = f"pointcloud_generation_failed: {e}"
            self.get_logger().error(f"❌ 点云生成失败:\n{traceback.format_exc()}")
            return response

        n_points = len(pcd_crop.points)
        self.get_logger().info(f"   点云生成完成: {n_points} 点 (裁剪后)")
        if n_points < 100:
            response.error_msg = "too_few_points_after_crop"
            self.get_logger().warn(f"❌ 点云点数过少 ({n_points})")
            return response

        # 3. GraspNet 推理
        t_inf_start = time.time()
        gg = self._run_graspnet(pcd_crop)
        t_inf = time.time() - t_inf_start
        self.get_logger().info(
            f"   GraspNet 推理完成: {len(gg)} 个候选, 耗时 {t_inf:.2f}s"
        )

        if len(gg) == 0:
            response.error_msg = "graspnet_returned_empty"
            return response

        # 4. score 过滤 (z-down 在 transform 后做)
        candidates_raw = self._filter_grasps(gg)
        self.get_logger().info(
            f"   score 过滤后: {len(candidates_raw)} 个候选 "
            f"(阈值 {self.score_threshold})"
        )
        # ── 三坐标系对比诊断 (cam 系, 米) ──
        # 对比 GraspNet grasp.translation vs 点云中心 vs YOLO bbox 中心
        # 用于定位 GraspNet xy 系统性偏移的根因
        if candidates_raw:
            g0 = candidates_raw[0]
            grasp_t = np.asarray(g0["translation"], dtype=np.float64)
            # 1) 点云中心 (裁剪+预处理后, cam 系)
            pcd_pts = np.asarray(pcd_crop.points, dtype=np.float64)
            pcd_center = pcd_pts.mean(axis=0) if len(pcd_pts) > 0 else np.full(3, np.nan)
            # 2) YOLO bbox 中心像素 -> organized 点云取三维坐标 (cam 系)
            bx1, by1, bx2, by2 = [int(round(v)) for v in bbox]
            bcx = (bx1 + bx2) // 2
            bcy = (by1 + by2) // 2
            bcy = max(0, min(bcy, cloud_organized.shape[0] - 1))
            bcx = max(0, min(bcx, cloud_organized.shape[1] - 1))
            yolo_center = np.asarray(cloud_organized[bcy, bcx], dtype=np.float64)
            # 3) bbox 中心原始深度 (mm, 校正前)
            raw_depth_mm = float(depth_img[bcy, bcx])
            self.get_logger().info("=" * 60)
            self.get_logger().info("📐 三坐标系对比 (cam 系, 米):")
            self.get_logger().info(f"   [A] GraspNet grasp.translation : ({grasp_t[0]:.4f}, {grasp_t[1]:.4f}, {grasp_t[2]:.4f})")
            self.get_logger().info(f"   [B] 点云中心 (pcd_crop mean)    : ({pcd_center[0]:.4f}, {pcd_center[1]:.4f}, {pcd_center[2]:.4f})")
            self.get_logger().info(f"   [C] YOLO bbox 中心 organized    : ({yolo_center[0]:.4f}, {yolo_center[1]:.4f}, {yolo_center[2]:.4f})  px=({bcx},{bcy})")
            self.get_logger().info(f"   [D] 原始深度图 bbox 中心深度    : {raw_depth_mm:.1f} mm (校正前)")
            diff_ab = grasp_t - pcd_center
            diff_ac = grasp_t - yolo_center
            self.get_logger().info(f"   [Δ A-B] grasp - 点云中心 : ({diff_ab[0]:+.4f}, {diff_ab[1]:+.4f}, {diff_ab[2]:+.4f})")
            self.get_logger().info(f"   [Δ A-C] grasp - YOLO中心 : ({diff_ac[0]:+.4f}, {diff_ac[1]:+.4f}, {diff_ac[2]:+.4f})")
            self.get_logger().info(f"   (若 Δ A-C 的 cam_z ≈ 物体半径, 说明 grasp 测的是物体中心而非表面)")
            self.get_logger().info("=" * 60)

        if not candidates_raw:
            response.error_msg = "no_grasps_passed_score_filter"
            return response

        # 5. 坐标变换 + z-down + workspace 过滤
        candidates_base = []
        for g in candidates_raw:
            result = self._transform_grasp_to_base(g)
            if result is not None:
                candidates_base.append({
                    **g,
                    **result,
                })
        self.get_logger().info(
            f"   z-down + workspace 过滤后: {len(candidates_base)} 个候选"
        )
        if not candidates_base:
            response.error_msg = "no_grasps_passed_filter"
            return response

        # ── 短轴对齐: 用 PCA 计算物体短轴, 重写候选 X 轴 (夹爪开合方向) ──
        # 使夹爪夹住物体较短方向 (手指从两条长边合拢夹窄腰)
        if self.align_gripper_to_short_axis:
            short_axis, long_axis, elong_ratio = self._compute_short_axis_base(
                pcd_crop
            )
            if short_axis is not None:
                self.get_logger().info(
                    f"   📏 PCA: 长轴=({long_axis[0]:.3f},{long_axis[1]:.3f}), "
                    f"短轴=({short_axis[0]:.3f},{short_axis[1]:.3f}), "
                    f"elongation_ratio={elong_ratio:.2f}"
                )
                if elong_ratio >= self.elongation_ratio_threshold:
                    rewritten = 0
                    for c in candidates_base:
                        Z = np.asarray(c["approach"], dtype=np.float64)
                        Z = Z / max(np.linalg.norm(Z), 1e-9)
                        # X = 短轴投影到 ⊥Z 平面
                        X = short_axis - np.dot(short_axis, Z) * Z
                        X = X / max(np.linalg.norm(X), 1e-9)
                        # Y = Z × X (右手系)
                        Y = np.cross(Z, X)
                        Y = Y / max(np.linalg.norm(Y), 1e-9)
                        c["rotation_matrix"] = np.column_stack([X, Y, Z])
                        rewritten += 1
                    self.get_logger().info(
                        f"   ✅ 短轴对齐已触发: 重写 {rewritten} 个候选的 X 轴 "
                        f"(夹爪开合方向对齐物体短轴)"
                    )
                else:
                    self.get_logger().info(
                        f"   ⏭️ 物体近圆形 (ratio={elong_ratio:.2f} < "
                        f"{self.elongation_ratio_threshold}), 保持 GraspNet 原姿态"
                    )
            else:
                self.get_logger().warn("   ⚠️ 短轴计算返回 None, 保持 GraspNet 原姿态")

        # 6. 取 top-N, 构造 Response
        candidates_base = candidates_base[:max_n]
        response.success = True
        response.best_score = float(candidates_base[0]["score"])
        response.depth_stamp.sec = int(depth_stamp_ns // 1_000_000_000)
        response.depth_stamp.nanosec = int(depth_stamp_ns % 1_000_000_000)

        for i, c in enumerate(candidates_base):
            gc = GraspCandidate()
            # PoseStamped
            ps = PoseStamped()
            ps.header.stamp = request.header.stamp
            ps.header.frame_id = self.base_frame_id
            ps.pose.position.x = float(c["position"][0])
            ps.pose.position.y = float(c["position"][1])
            ps.pose.position.z = float(c["position"][2])
            # rotation_matrix -> quaternion
            qx, qy, qz, qw = self._rotmat_to_quat(c["rotation_matrix"])
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw
            gc.grasp_pose = ps
            gc.score = float(c["score"])
            gc.approach_angle_deg = float(c["approach_angle_deg"])
            response.candidates.append(gc)

        # 7. 发布 RViz Marker
        self._publish_markers(candidates_base, request.header.stamp)

        t_total = time.time() - t_start
        self.get_logger().info(
            f"✅ 返回 {len(response.candidates)} 个候选, "
            f"best_score={response.best_score:.3f}, 总耗时 {t_total:.2f}s"
        )
        return response

    # ========================================================================
    # RViz 可视化
    # ========================================================================
    def _publish_markers(self, candidates, stamp):
        """发布 MarkerArray, 每个 candidate 一个 ARROW"""
        marker_array = MarkerArray()
        # 先发一个 DELETEALL 清除旧 marker
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for i, c in enumerate(candidates):
            m = Marker()
            m.header.stamp = stamp
            m.header.frame_id = self.base_frame_id
            m.ns = "grasp_poses"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            # 起点 = 抓取位置
            start = c["position"]
            # 终点 = start + approach * 0.1 (10cm)
            approach = c["approach"]
            end = start + approach * 0.1
            m.points = [
                Point(x=float(start[0]), y=float(start[1]), z=float(start[2])),
                Point(x=float(end[0]), y=float(end[1]), z=float(end[2])),
            ]
            # 颜色按 score
            score = c["score"]
            color = ColorRGBA()
            color.a = 0.9
            if score > 0.7:
                color.r, color.g, color.b = 0.0, 1.0, 0.0  # 绿
            elif score > 0.5:
                color.r, color.g, color.b = 1.0, 1.0, 0.0  # 黄
            else:
                color.r, color.g, color.b = 1.0, 0.0, 0.0  # 红
            m.color = color
            m.scale.x = 0.005  # shaft diameter
            m.scale.y = 0.012  # head diameter
            m.lifetime.sec = 5
            marker_array.markers.append(m)

        self._marker_pub.publish(marker_array)

    @staticmethod
    def _rotmat_to_quat(R):
        """3x3 旋转矩阵 -> 四元数 (x, y, z, w)"""
        # 使用 Shepperd 方法
        m = R
        trace = m[0, 0] + m[1, 1] + m[2, 2]
        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        # 归一化
        n = math.sqrt(x * x + y * y + z * z + w * w)
        if n > 1e-9:
            x, y, z, w = x / n, y / n, z / n, w / n
        return float(x), float(y), float(z), float(w)


def main(args=None):
    rclpy.init(args=args)
    node = GraspPoseNode()
    try:
        # MultiThreadedExecutor 让 service 回调不阻塞订阅回调
        # 但 GraspNet 推理是同步的, 服务回调期间订阅会被阻塞 (cb_service 独占组)
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
