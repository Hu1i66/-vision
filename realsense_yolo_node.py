#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import Marker
from tf2_ros import TransformBroadcaster
import cv2
import numpy as np
import json
import math
import tf_transformations


class ObjectTrack:
    """单个物体 track 的 EMA (指数移动平均) 滤波状态。

    用于平滑 YOLO 检测结果, 消除像素 bbox、深度、坐标的帧间抖动。
    每个 track 绑定一个类别 + 空间位置 (像素中心), 通过最近邻匹配关联帧间检测。
    """

    def __init__(self, track_id, class_name, obs, frame_count, alpha=0.25):
        self.track_id = track_id
        self.class_name = class_name
        self.alpha = alpha  # EMA 系数: 新观测权重, 越小越平滑
        # 上一帧像素中心 (用于匹配)
        self.last_center_u = obs['center_u']
        self.last_center_v = obs['center_v']
        # 滤波状态 (初始值 = 第一次观测)
        self.s_x1 = float(obs['x1'])
        self.s_y1 = float(obs['y1'])
        self.s_x2 = float(obs['x2'])
        self.s_y2 = float(obs['y2'])
        self.s_depth = obs.get('depth')        # float or None
        self.s_cam = list(obs['camera_coords']) if obs.get('camera_coords') is not None else None
        self.s_ee = list(obs['ee_coords']) if obs.get('ee_coords') is not None else None
        self.s_base = list(obs['base_coords']) if obs.get('base_coords') is not None else None
        # 元数据
        self.last_update_frame = frame_count
        self.miss_count = 0
        self.confidence = obs.get('conf', 0.0)
        self.last_obs = obs  # 保留最新原始观测 (调试用)

    def update(self, obs, frame_count, alpha=None):
        """用新观测做 EMA 更新。alpha=None 时用 self.alpha。
        深度失败的帧不更新 s_depth/s_cam/s_ee/s_base (保留上一帧滤波值)。"""
        a = alpha if alpha is not None else self.alpha
        self.last_center_u = obs['center_u']
        self.last_center_v = obs['center_v']
        # bbox EMA
        self.s_x1 = a * float(obs['x1']) + (1 - a) * self.s_x1
        self.s_y1 = a * float(obs['y1']) + (1 - a) * self.s_y1
        self.s_x2 = a * float(obs['x2']) + (1 - a) * self.s_x2
        self.s_y2 = a * float(obs['y2']) + (1 - a) * self.s_y2
        # 深度 EMA (深度失败时保留旧值, 不更新)
        d = obs.get('depth')
        if d is not None:
            if self.s_depth is None:
                self.s_depth = d
            else:
                self.s_depth = a * d + (1 - a) * self.s_depth
        # 相机坐标 EMA
        cam = obs.get('camera_coords')
        if cam is not None:
            if self.s_cam is None:
                self.s_cam = list(cam)
            else:
                self.s_cam = [a * cam[i] + (1 - a) * self.s_cam[i] for i in range(3)]
        # 末端坐标 EMA
        ee = obs.get('ee_coords')
        if ee is not None:
            if self.s_ee is None:
                self.s_ee = list(ee)
            else:
                self.s_ee = [a * ee[i] + (1 - a) * self.s_ee[i] for i in range(3)]
        # 基座坐标 EMA
        base = obs.get('base_coords')
        if base is not None:
            if self.s_base is None:
                self.s_base = list(base)
            else:
                self.s_base = [a * base[i] + (1 - a) * self.s_base[i] for i in range(3)]
        # 元数据
        self.last_update_frame = frame_count
        self.miss_count = 0
        self.confidence = obs.get('conf', self.confidence)
        self.last_obs = obs


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")

        # ================== 模型配置 ==================
        # 新权重支持 7 类水果检测 (green apple / apple / honey peach / lemon / orange / pear / strawberry)
        self.MODEL_PATH = "/home/lxf/orange_dataset/best (2).pt"
        # 置信度阈值: 与另一台电脑 (model(frame) 默认 conf=0.25) 保持一致。
        # 之前 0.8 太高, 过滤掉大量有效检测导致识别率低。
        self.CONF_THRESHOLD = 0.25

        # ================== 显示模式 ==================
        # True: standalone 模式，弹出 OpenCV 窗口；False: GUI 内嵌模式，仅发布 ROS topic
        self.show_gui_window = bool(self.declare_parameter("show_gui_window", True).value)

        # ================== 已知物体真实尺寸 (单目深度反推用) ==================
        # 每个类别的真实物理尺寸 (单位: 米)。来源: 用户实测。
        # length=长(对应像素宽度), width=宽(深度方向), height=高(对应像素高度)
        # 单目深度估计用 length (像素宽度方向) 和 height (像素高度方向) 反推 depth。
        self.OBJECT_DIMENSIONS = {
            "green apple": {"length_m": 0.0836, "width_m": 0.0836, "height_m": 0.0715},
            "apple":       {"length_m": 0.076, "width_m": 0.076, "height_m": 0.065},
            "honey peach": {"length_m": 0.075, "width_m": 0.076, "height_m": 0.065},
            "lemon":       {"length_m": 0.080, "width_m": 0.054, "height_m": 0.0648},
            "orange":      {"length_m": 0.069, "width_m": 0.069, "height_m": 0.055},
            "pear":        {"length_m": 0.090, "width_m": 0.075, "height_m": 0.074},
            "strawberry":  {"length_m": 0.090, "width_m": 0.060, "height_m": 0.046},
        }
        # 默认尺寸 (兜底: 类别未在字典中时使用, 用 orange 的尺寸)
        self.known_object_width_m  = float(self.declare_parameter("known_object_width_m",  0.069).value)
        self.known_object_height_m = float(self.declare_parameter("known_object_height_m", 0.055).value)

        # ================== 相机参数 (RealSense D455) ==================
        self.fx = 378.394659861614
        self.fy = 379.366916262423
        self.cx = 330.140969430714
        self.cy = 246.095530649072
        self.skew = 1.25667775035477

        # 完整相机内参矩阵 (3x3)
        self.camera_matrix = np.array([
            [self.fx, self.skew, self.cx],
            [0,       self.fy,   self.cy],
            [0,       0,         1]
        ], dtype=np.float64)

        # 畸变系数: [k1, k2, p1, p2, k3]
        self.dist_coeffs = np.array([
            -0.0579748391767018,      # k1
            0.104170020380745,        # k2
            -0.000760277417774888,    # p1
            0.000473717313218756,     # p2
            -0.0946382536306512       # k3
        ], dtype=np.float64)

        # ── camera_info 实时内参 (订阅 /camera/camera/color/camera_info 后覆盖上方硬编码) ──
        # D455 出厂标定的内参通过 camera_info 话题发布, 比硬编码值更准
        # (fx≈384.8 vs 硬编码 378.4, ~1.7% 差异在 0.5m 处产生 ~8.5mm 位置误差)
        # 硬编码值保留作为 camera_info 未到达时的 fallback
        self._cam_info_received = False
        
        # ================== 手眼标定数据（相机到机械臂基座, 眼在手外）==================
        # 来源: /home/lxf/handeye/result/2026-07-26_16-17-46_calibration.json
        # eye-to-hand: 相机固定在传送带上方, T_cam_to_base 是常量 (不依赖 TCP 位姿)
        # 平移向量 (单位: 米)
        self.camera_to_base_translation = np.array([
            0.6732749433043899,   # X
            0.010662122461489898, # Y
            0.597133855461562     # Z
        ])

        # 四元数 (x, y, z, w)
        self.camera_to_base_quat = np.array([
            -0.6934151560947895,   # qx
            -0.7119098057466932,   # qy
            0.08845804637870601,   # qz
            0.06734258541670725    # qw
        ])

        # RPY 角度 (弧度)
        self.camera_to_base_rpy = np.array([
            -2.9203729063228363,   # roll
            0.026795812790156897,  # pitch
            1.6000918271309976     # yaw
        ])

        # ================== 构建完整的变换矩阵 ==================
        # 方法1：使用四元数构建旋转矩阵
        self.rotation_matrix_cam_to_base = self.quaternion_to_rotation_matrix(
            self.camera_to_base_quat
        )

        # 方法2：使用 RPY 角度构建旋转矩阵（验证用）
        self.rotation_matrix_cam_to_base_rpy = self.rpy_to_rotation_matrix(
            self.camera_to_base_rpy
        )

        # 构建 4x4 齐次变换矩阵 (相机坐标系 -> 机械臂基坐标系, eye-to-hand 常量)
        self.transform_cam_to_base = np.eye(4)
        self.transform_cam_to_base[:3, :3] = self.rotation_matrix_cam_to_base
        self.transform_cam_to_base[:3, 3] = self.camera_to_base_translation

        # 计算基座到相机的逆变换（用于验证）
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
        self.get_logger().info(f"   RPY: roll={math.degrees(self.camera_to_base_rpy[0]):.2f}°")
        self.get_logger().info(f"         pitch={math.degrees(self.camera_to_base_rpy[1]):.2f}°")
        self.get_logger().info(f"         yaw={math.degrees(self.camera_to_base_rpy[2]):.2f}°")
        self.get_logger().info("=" * 60)

        # ================== eye-to-hand: 无需机械臂位姿订阅 ==================
        # 相机固定, T_cam_to_base 是常量, 不再订阅 /feedback/tcp_pose
        # RViz 可视化用的基座坐标系 frame_id
        self.base_frame_id = self.declare_parameter(
            "base_frame_id",
            "base_link"
        ).value
        
        # ================== 初始化 ==================
        self.latest_rgb = None
        self.latest_rgb_stamp = None
        self.rgb_ready = False
        self.frame_count = 0

        # ================== 滤波参数 (EMA + 空间关联) ==================
        # filter_alpha: EMA 新观测权重, 越小越平滑 (0.1=强滤波, 0.5=弱滤波, 0.0=完全冻结)
        # filter_match_dist_px: 帧间匹配阈值 (像素), 同类物体中心距离 < 此值视为同一物体
        # filter_max_misses: 连续多少帧未匹配到则删除 track
        self.filter_alpha = float(self.declare_parameter("filter_alpha", 0.25).value)
        self.filter_match_dist_px = float(self.declare_parameter("filter_match_dist_px", 80.0).value)
        self.filter_max_misses = int(self.declare_parameter("filter_max_misses", 10).value)
        self._tracks = {}        # {track_id: ObjectTrack}
        self._next_track_id = 0

        # ================== 深度相机 (D455) 配置 ==================
        # 深度相机开关 (默认开启, 失败时自动回退到单目)
        self.use_depth_camera = self.declare_parameter('use_depth_camera', True).value
        self._depth_fallback_count = 0  # 统计回退到单目的次数
        self._depth_total_count = 0    # 深度估计总次数 (D455尝试次数, 用于回退比率)

        # ── D455 深度偏差校正 (单位: 米) ──
        # eye-to-hand: 相机在可靠深度范围 (0.5-0.8m), 无需近距校正, 默认 0.0
        # 保留参数供未来微调: 正值=减深度=物体抬高
        self.depth_offset_m = float(self.declare_parameter('depth_offset_m', 0.0).value)
        self._depth_diag_logged = False  # 仅首帧打印诊断

        # ── eye-to-hand X/Y 标定偏差补偿 (单位: 米) ──
        # 现象: base_position_m 的 X/Y 偏大 2-3cm → GUI 显示和抓取都偏
        # 补偿: 加到 base 坐标 (负值=减去=修正偏大, 默认 -0.025 = 减 2.5cm)
        # 调参: ros2 param set /object_detector x_offset_m -0.020
        #       ros2 param set /object_detector y_offset_m -0.020
        self.x_offset_m = float(self.declare_parameter('x_offset_m', -0.045).value)
        self.y_offset_m = float(self.declare_parameter('y_offset_m', -0.010).value)
        # Z 补偿: 正值=抬高 (防下探过多撞台面), 默认 +0.030 = 加 3cm
        # 调参: ros2 param set /object_detector z_offset_m 0.025
        self.z_offset_m = float(self.declare_parameter('z_offset_m', 0.030).value)

        # 统计信息
        self.detection_count = 0
        self.status_printed = False
        
        # ================== 创建发布/订阅 ==================
        self.pub_object_pose = self.create_publisher(Float64MultiArray, "/object_3d_position", 10)
        self.pub_detection_info = self.create_publisher(String, "/detection_info", 10)

        # ── 发布带 YOLO 标注框的压缩画面，供 GUI 直接显示 ──
        self.pub_annotated = self.create_publisher(CompressedImage, "/yolo/annotated_image/compressed", 1)

        # RViz 可视化
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_object_marker = self.create_publisher(Marker, "/object_marker", 10)
        self.pub_object_pose_stamped = self.create_publisher(PoseStamped, "/object_pose", 10)

        # eye-to-hand: 无需机械臂位姿订阅, 相机变换是常量

        # 纯RGB单目模式：只订阅彩色图像，深度由已知物体尺寸反推
        self.sub_rgb = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_cb, 10)

        # D455 对齐深度图订阅 (与 RGB 像素坐标严格对齐)
        self.sub_depth = self.create_subscription(
            Image,
            "/camera/camera/aligned_depth_to_color/image_raw",
            self.depth_cb,
            10
        )
        self.latest_depth = None           # uint16 ndarray (mm)
        self.latest_depth_stamp = 0.0      # 时间戳 (秒)

        # ── 订阅 camera_info: 实时获取 D455 出厂内参 (替代上方硬编码) ──
        self.sub_cam_info = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_cb,
            10
        )

        # ==================== 曝光调节相关代码 ====================
        # 注意: 本节点通过 ROS2 订阅图像, 不直接持有 RealSense pipeline。
        # 曝光调节通过 ros2 param set 命令修改 realsense-ros 驱动的动态参数实现,
        # 等效于用户给的 pyrealsense2 color_sensor.set_option(rs.option.exposure)。
        # realsense-ros 参数命名 (见 rs_launch.py L39): rgb_camera.enable_auto_exposure
        self.camera_param_node = self.declare_parameter(
            "camera_param_node", "/camera/camera"
        ).value
        self.auto_exposure_param_name = self.declare_parameter(
            "auto_exposure_param_name", "rgb_camera.enable_auto_exposure"
        ).value
        self.exposure_param_name = self.declare_parameter(
            "exposure_param_name", "rgb_camera.exposure"
        ).value

        # RealSense 曝光范围 (微秒, D435/D455 典型值 1-10000)
        self.exposure_min = 1
        self.exposure_max = 10000
        self.current_exposure = 100
        self.auto_exposure = True

        # 订阅 GUI 发来的曝光控制命令 (JSON: {"auto": true} 或 {"auto": false, "value": 5000})
        self.create_subscription(String, "/camera/exposure_ctrl", self._exposure_ctrl_cb, 10)

        self.get_logger().info("正在加载 YOLO 模型...")
        self.get_logger().info(f"📐 单目估计: 已知物体尺寸 W={self.known_object_width_m:.3f}m H={self.known_object_height_m:.3f}m")
        from ultralytics import YOLO
        self.model = YOLO(self.MODEL_PATH)
        self.get_logger().info(f"✅ 模型加载成功，类别: {self.model.names}")

        # 定时器
        self.create_timer(0.1, self.detect_and_publish)
        self.status_timer = self.create_timer(5.0, self.print_status)
        self._depth_status_timer = self.create_timer(5.0, self._depth_status_log)

        # OpenCV 窗口 (仅 standalone 模式)
        if self.show_gui_window:
            cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Object Detection", 960, 540)

        self.get_logger().info("✅ 节点启动，等待相机数据...")
        self.get_logger().info(f"🔍 检测阈值: {self.CONF_THRESHOLD}")
        self.get_logger().info(f"📷 眼在手外 (eye-to-hand): 相机变换为常量, 无需 TCP 位姿订阅")
        self.get_logger().info(f"🖼️  RViz TF 基座 frame: {self.base_frame_id}")

        # 默认启用自动曝光 (相机未上线时 ros2 param set 会静默失败, 不影响节点启动)
        self.set_exposure(True)

    # ==================== 滤波: track 匹配与清理 ====================
    def _find_matching_track(self, class_name, center_u, center_v, exclude_ids=None):
        """在现有 tracks 中找同类且像素中心距离最近的 track (贪心匹配)。
        exclude_ids: 本帧已匹配过的 track_id 集合, 避免重复匹配。
        返回 ObjectTrack 或 None。"""
        exclude_ids = exclude_ids or set()
        best_track = None
        best_dist = self.filter_match_dist_px
        for track in self._tracks.values():
            if track.track_id in exclude_ids:
                continue
            if track.class_name != class_name:
                continue
            dx = track.last_center_u - center_u
            dy = track.last_center_v - center_v
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_track = track
        return best_track

    def _cleanup_stale_tracks(self, frame_count):
        """删除连续 filter_max_misses 帧未匹配的 track, 并对未匹配 track miss_count++。"""
        stale_ids = [
            tid for tid, t in self._tracks.items()
            if t.miss_count >= self.filter_max_misses
        ]
        for tid in stale_ids:
            del self._tracks[tid]
        for t in self._tracks.values():
            if t.last_update_frame < frame_count:
                t.miss_count += 1

    # ==================== 物体尺寸查询 ====================
    def _get_class_dimensions(self, class_name):
        """按类别名查询物体真实尺寸 (length_m, height_m)。
        类别名做小写匹配, 找不到时回落到默认 known_object_*_m。"""
        if not class_name:
            return self.known_object_width_m, self.known_object_height_m
        key = class_name.strip().lower()
        dim = self.OBJECT_DIMENSIONS.get(key)
        if dim is None:
            # 模糊匹配: 类别名包含字典 key (例如 "green apple" 包含 "apple")
            for k, v in self.OBJECT_DIMENSIONS.items():
                if k in key or key in k:
                    dim = v
                    break
        if dim is None:
            self.get_logger().warning(
                f"未注册的类别 '{class_name}', 回落到默认尺寸 "
                f"W={self.known_object_width_m:.3f}m H={self.known_object_height_m:.3f}m",
                throttle_duration_sec=5.0
            )
            return self.known_object_width_m, self.known_object_height_m
        return dim["length_m"], dim["height_m"]

    # ==================== 曝光调节 ====================
    def _exposure_ctrl_cb(self, msg):
        """接收 GUI 发来的曝光控制命令。
        JSON 格式: {"auto": true} 或 {"auto": false, "value": 5000}
        """
        try:
            cmd = json.loads(msg.data)
            auto = bool(cmd.get("auto", True))
            value = cmd.get("value", None)
            self.set_exposure(auto, value if value is None else int(value))
        except Exception as e:
            self.get_logger().error(f"解析曝光控制命令失败: {e} (raw: {msg.data[:100]})")

    def set_exposure(self, auto, value=None):
        """通过 ros2 param set 调节 realsense-ros 相机驱动的曝光。
        等效于 pyrealsense2: color_sensor.set_option(rs.option.enable_auto_exposure, auto)
                            color_sensor.set_option(rs.option.exposure, value)
        使用 subprocess 异步调用, 不阻塞主检测循环。
        """
        import subprocess
        self.auto_exposure = auto
        ros_setup = "source /opt/ros/humble/setup.bash"
        if auto:
            # 启用自动曝光
            cmd = (f'{ros_setup} && ros2 param set {self.camera_param_node} '
                   f'{self.auto_exposure_param_name} true')
            subprocess.Popen(['bash', '-c', cmd],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.get_logger().info("🔄 已切换为自动曝光")
        else:
            # 关闭自动曝光
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
                    f"(范围: {int(self.exposure_min)} - {int(self.exposure_max)})"
                )

    def quaternion_to_rotation_matrix(self, quat):
        """
        将四元数转换为旋转矩阵
        quat: [qx, qy, qz, qw]
        """
        qx, qy, qz, qw = quat
        
        rotation_matrix = np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])
        
        return rotation_matrix
    
    def rpy_to_rotation_matrix(self, rpy):
        """
        将 RPY 角度（弧度）转换为旋转矩阵
        rpy: [roll, pitch, yaw]
        """
        roll, pitch, yaw = rpy
        
        R_x = np.array([
            [1, 0, 0],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll), math.cos(roll)]
        ])
        
        R_y = np.array([
            [math.cos(pitch), 0, math.sin(pitch)],
            [0, 1, 0],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])
        
        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        rotation_matrix = R_z @ R_y @ R_x
        return rotation_matrix
    
    def pose_stamped_to_dict(self, pose_msg):
        """将 PoseStamped 转为可发布的字典。"""
        if pose_msg is None:
            return None
        return {
            "header_frame_id": pose_msg.header.frame_id,
            "position": {
                "x": round(pose_msg.pose.position.x, 4),
                "y": round(pose_msg.pose.position.y, 4),
                "z": round(pose_msg.pose.position.z, 4)
            },
            "orientation": {
                "x": round(pose_msg.pose.orientation.x, 6),
                "y": round(pose_msg.pose.orientation.y, 6),
                "z": round(pose_msg.pose.orientation.z, 6),
                "w": round(pose_msg.pose.orientation.w, 6)
            }
        }
    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def print_status(self):
        """打印状态信息"""
        if not self.status_printed:
            self.status_printed = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("📊 状态统计:")
            self.get_logger().info(f"  - RGB 数据: {'就绪' if self.rgb_ready else '等待中'}")
            self.get_logger().info(f"  - 眼在手外: 相机变换为常量 (无需机械臂位姿)")
            self.get_logger().info(f"  - 已处理帧数: {self.frame_count}")
            self.get_logger().info(f"  - 检测到物体: {self.detection_count} 次")
            self.get_logger().info("=" * 60)
            self.status_timer.cancel()

    def _depth_status_log(self):
        """周期诊断 (每5s): D455 深度图接收状态 + 单目回退比率.
        结论性区分两种故障:
          (a) latest_depth=None → 深度话题未收到 → 全程单目 (直径≈预设/距离抖动/质心z=None→撞台面)
          (b) 话题活着但逐物体回退 → bbox中心5x5无效像素<3 或 深度超出0.2~3.0m
        """
        if not self.use_depth_camera:
            return
        if self.latest_depth is None:
            self.get_logger().error(
                "❌ D455 深度图未收到 (latest_depth=None)! "
                "话题 /camera/camera/aligned_depth_to_color/image_raw 无数据 → 全程单目 "
                "(用预设尺寸: 直径≈预设, 距离抖动, 质心z=None→夹爪撞台面). "
                "修复: 相机以 align_depth.enable:=true 启动, 并确认 ros2 topic list 含 aligned_depth")
            return
        total = self._depth_total_count
        fb = self._depth_fallback_count
        if total > 0:
            ok = total - fb
            ratio = fb / total * 100.0
            if fb > 0:
                self.get_logger().warn(
                    f"⚠️ D455 深度: {ok}/{total} 成功, {fb} 回退单目 ({ratio:.0f}%). "
                    f"回退时用预设尺寸 (直径≈预设, 距离抖动). "
                    f"可能原因: bbox中心5x5无效像素<3 或 深度超出0.2~3.0m")
            else:
                self.get_logger().info(f"✅ D455 深度: {ok}/{total} 全部成功 (无单目回退)")
        # 重置周期计数
        self._depth_total_count = 0
        self._depth_fallback_count = 0

    def rgb_cb(self, msg):
        """接收 RGB 图像"""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.latest_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_rgb_stamp = self._stamp_to_sec(msg.header.stamp)

            if not self.rgb_ready:
                self.rgb_ready = True
                self.get_logger().info(f"✅ RGB 就绪: {msg.width}x{msg.height}")
        except Exception as e:
            self.get_logger().error(f"RGB 转换错误: {e}")

    def depth_cb(self, msg):
        """接收 D455 对齐深度图 (uint16, mm). 像素坐标与 RGB 严格对齐."""
        try:
            self.latest_depth = np.frombuffer(
                msg.data, dtype=np.uint16
            ).reshape(msg.height, msg.width).copy()
            self.latest_depth_stamp = self._stamp_to_sec(msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"深度图解析失败: {e}")
            self.latest_depth = None
            
    def transform_camera_to_base(self, camera_coords):
        """相机坐标 → 基座坐标 (eye-to-hand: 直接用常量变换矩阵)

        eye-to-hand 配置下相机固定, T_cam_to_base 是常量, 不再需要
        相机→末端→基座的两阶段变换, 也无需 TCP 位姿订阅.
        """
        if camera_coords is None:
            return None, None
        point_homogeneous = np.append(camera_coords, 1)
        base_coords = self.transform_cam_to_base @ point_homogeneous
        # X/Y 标定偏差补偿 (仅修正 main detection 位置, 不影响 PCA/质心 z 计算)
        base_coords[0] += self.x_offset_m
        base_coords[1] += self.y_offset_m
        base_coords[2] += self.z_offset_m  # Z 补偿 (防下探过多)
        return base_coords[:3], None  # ee_coords 不再适用, 返回 None

    def get_robot_current_position(self):
        """eye-to-hand: 不再跟踪机械臂位姿, 返回 None"""
        return None
    
    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        if point1 is None or point2 is None:
            return None
        return np.linalg.norm(point1 - point2)

    def estimate_object_volume(self, width, height, shape="sphere"):
        """估计物体体积"""
        if shape == "sphere":
            diameter = (width + height) / 2
            radius = diameter / 2
            volume = (4/3) * math.pi * (radius ** 3)
        elif shape == "cube":
            volume = width * height * width
        else:
            volume = width * height * width
        return volume

    # ================== 单目RGB估计方法 ==================

    def monocular_depth_from_bbox(self, box, class_name=None):
        """
        利用已知物体真实宽度，通过针孔模型反推深度。
        原理: depth = fx * real_width / pixel_width
        返回: depth_m (float) 或 None
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        pixel_width = float(x2 - x1)
        pixel_height = float(y2 - y1)

        if pixel_width <= 0 or pixel_height <= 0:
            return None

        # 按检测到的类别查询真实尺寸 (未注册类别回落到默认 known_object_*_m)
        real_width, real_height = self._get_class_dimensions(class_name)
        # 用fx和已知宽度反推；同时用fy+已知高度做交叉验证
        depth_w = self.fx * real_width / pixel_width
        depth_h = self.fy * real_height / pixel_height
        depth = (depth_w + depth_h) / 2.0

        if depth < 0.05 or depth > 10.0:
            return None

        return depth

    def depth_from_camera(self, box):
        """从 D455 对齐深度图提取物体深度 (米).

        策略: 取 bbox 中心 5x5 像素区域的中位数.
        - 用中位数而非均值: 抗噪、抗无效像素
        - 用中心小区域而非整个 bbox: 避免 bbox 松散时混入背景深度
        - 水果类物体近似球/椭球, bbox 中心通常落在物体表面

        Args:
            box: [x1, y1, x2, y2] 像素坐标

        Returns:
            depth_m (float) 或 None (无有效深度)
        """
        if self.latest_depth is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in box]
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        h, w = self.latest_depth.shape
        x_start = max(0, cx - 2)
        x_end = min(w, cx + 3)
        y_start = max(0, cy - 2)
        y_end = min(h, cy + 3)

        region = self.latest_depth[y_start:y_end, x_start:x_end]
        valid = region[region > 0]

        if len(valid) < 3:
            return None

        depth_mm = float(np.median(valid))
        depth_m = depth_mm / 1000.0

        if depth_m < 0.2 or depth_m > 3.0:
            return None

        # ── D455 近距离偏差校正 ──
        # 减去 depth_offset_m 修正 D455 在 <0.4m 近距离的系统性偏大
        raw_depth_m = depth_m
        if self.depth_offset_m != 0.0:
            depth_m = depth_m - self.depth_offset_m
            if depth_m < 0.1:
                depth_m = 0.1  # 防止过度校正

        # ── 诊断日志: D455 原始深度 vs 单目估算深度 (仅首帧) ──
        if not self._depth_diag_logged:
            self._depth_diag_logged = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("📐 D455 深度诊断 (首帧):")
            self.get_logger().info(f"   D455 原始深度 (cam_z): {raw_depth_m:.3f}m")
            self.get_logger().info(f"   校正量 depth_offset_m: {self.depth_offset_m:.3f}m")
            self.get_logger().info(f"   校正后深度 (cam_z): {depth_m:.3f}m")
            self.get_logger().info(f"   (若 base_z 仍偏低, 调大 depth_offset_m; 偏高则调小)")
            self.get_logger().info("=" * 60)

        return depth_m

    def monocular_pixel_to_camera_coords(self, u, v, depth):
        """
        单目模式: 用估算深度将像素坐标转为相机3D坐标
        u, v: 像素坐标
        depth: 估算深度 (米)
        """
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return np.array([x, y, z])

    def publish_object_visualization(self, camera_coords, base_coords):
        """将检测结果以 TF + Marker + PoseStamped 发布到 RViz"""
        now = self.get_clock().now().to_msg()

        # 选择发布位置：优先基座坐标，否则用相机坐标
        if base_coords is not None:
            pos_x, pos_y, pos_z = base_coords
            parent_frame = self.base_frame_id  # base_coords 已在基座坐标系下
        else:
            pos_x, pos_y, pos_z = camera_coords
            parent_frame = "camera_frame"      # camera_coords 在相机坐标系下

        # --- TF ---
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = parent_frame
        t.child_frame_id = "detected_object"
        t.transform.translation.x = pos_x
        t.transform.translation.y = pos_y
        t.transform.translation.z = pos_z
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        # --- Marker (绿色半透明球体) ---
        marker = Marker()
        marker.header.stamp = now
        marker.header.frame_id = parent_frame
        marker.ns = "detected_objects"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = pos_x
        marker.pose.position.y = pos_y
        marker.pose.position.z = pos_z
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.known_object_width_m
        marker.scale.y = self.known_object_height_m
        marker.scale.z = self.known_object_height_m
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.5
        marker.lifetime.sec = 1
        self.pub_object_marker.publish(marker)

        # --- PoseStamped ---
        ps = PoseStamped()
        ps.header.stamp = now
        ps.header.frame_id = parent_frame
        ps.pose.position.x = pos_x
        ps.pose.position.y = pos_y
        ps.pose.position.z = pos_z
        ps.pose.orientation.w = 1.0
        self.pub_object_pose_stamped.publish(ps)

    def camera_info_cb(self, msg):
        """从 /camera/camera/color/camera_info 实时更新内参 (替代硬编码).

        D455 出厂标定的内参通过此话题发布。仅首帧更新 (内参不变),
        覆盖 __init__ 中的硬编码 fx/fy/cx/cy/camera_matrix/dist_coeffs。
        """
        if getattr(self, '_cam_info_received', False):
            return  # 仅首帧更新, 避免每帧重复解析
        try:
            k = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            new_fx = float(k[0, 0])
            new_fy = float(k[1, 1])
            new_cx = float(k[0, 2])
            new_cy = float(k[1, 2])
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

    def _compute_grasp_orientation(self, bbox_xyxy, depth_mm, object_base_z,
                                   margin_ratio=0.3, elongation_threshold=1.5,
                                   height_band=0.03, min_points=30):
        """从 bbox 内 D455 深度点云用 PCA 计算物体短轴, 生成夹爪朝向 (顶视 + 短轴对齐)。

        坐标系约定 (与 auto_sorting_action.py 一致):
          - 夹爪 Z 轴 = 接近方向 (朝下 [0,0,-1])
          - 夹爪 X 轴 = 开合方向 (对齐物体短轴, 手指从两条长边合拢夹窄腰)
          - 夹爪 Y 轴 = 手指长度方向 (Z × X)

        Args:
            bbox_xyxy: [x1,y1,x2,y2] 像素坐标 (滤波后)
            depth_mm: D455 对齐深度图 (uint16, mm) = self.latest_depth
            object_base_z: 物体在 base 系的 z (滤波后), 用于高度带通剔除传送带/背景
        Returns:
            (quat_list, elongation_ratio): quat=[qx,qy,qz,qw] 或 None (圆物体/失败)
        """
        if depth_mm is None:
            return None, None, None
        # eye-to-hand: T_cam_to_base 是常量, 不再依赖 robot_current_pose
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            h_img, w_img = depth_mm.shape
            bw = x2 - x1
            bh = y2 - y1
            mx = int(bw * margin_ratio)
            my = int(bh * margin_ratio)
            x1m = max(0, x1 - mx)
            y1m = max(0, y1 - my)
            x2m = min(w_img, x2 + mx)
            y2m = min(h_img, y2 + my)
            sub = depth_mm[y1m:y2m, x1m:x2m]
            if sub.size == 0:
                return None, None, None
            valid_mask = sub > 0
            ys, xs = np.nonzero(valid_mask)
            if len(xs) < min_points:
                return None, None, None
            depths_m = sub[ys, xs].astype(np.float64) / 1000.0
            # D455 深度偏移修正 (eye-to-hand 下默认 0.0)
            if self.depth_offset_m != 0.0:
                depths_m = depths_m - self.depth_offset_m
                depths_m = np.maximum(depths_m, 0.1)
            # 全局图像像素坐标
            us = xs + x1m
            vs = ys + y1m
            # 反投影到相机坐标 (用 camera_info 内参)
            cam_x = (us - self.cx) * depths_m / self.fx
            cam_y = (vs - self.cy) * depths_m / self.fy
            cam_z = depths_m
            # eye-to-hand: 直接变换到 base 系 (常量矩阵, 无需 TCP 位姿)
            cam_pts = np.stack([cam_x, cam_y, cam_z, np.ones_like(cam_x)], axis=0)  # 4xN
            base_pts = (self.transform_cam_to_base @ cam_pts)[:3, :]  # 3xN
            base_pts[2, :] += self.z_offset_m  # Z 补偿 (与主检测路径一致)
            bz = base_pts[2, :]
            # ── 质心 z 估算 (不依赖预设尺寸): 物体层点云 z 范围 ──
            # object_base_z 是表面 z (bbox 中心), 过滤远背景 (z < 表面-0.12),
            # 取物体层 z 的 [25%, 90%] 分位作为 [底, 顶], 高度=顶-底
            # 质心 = 表面 - height/2, 限制 height/2 <= 0.025
            # (防 z_bot 含背景桌面/D455近距偏差导致质心过低 → 撞台面)
            obj_layer_mask = bz > (object_base_z - 0.12)
            bz_obj = bz[obj_layer_mask]
            if len(bz_obj) >= 10:
                _z_top = float(np.percentile(bz_obj, 90))
                _z_bot = float(np.percentile(bz_obj, 25))
                _height = max(0.0, _z_top - _z_bot)
                _delta = min(_height / 2.0, 0.025)
                centroid_z = float(object_base_z) - _delta
            else:
                centroid_z = float(object_base_z)
            # 高度带通: 保留 base_z 在 [object_base_z ± height_band] 内的点
            band_mask = (bz >= object_base_z - height_band) & (bz <= object_base_z + height_band)
            bx = base_pts[0, band_mask]
            by = base_pts[1, band_mask]
            if len(bx) < min_points:
                return None, None, None
            # PCA on XY 分量 (2xN)
            pts_xy = np.stack([bx, by], axis=0)  # 2xN
            mean = pts_xy.mean(axis=1, keepdims=True)
            centered = pts_xy - mean
            cov = np.cov(centered)  # 2x2, rowvar=True
            eigvals, eigvecs = np.linalg.eigh(cov)  # 升序: eigvals[0] <= eigvals[1]
            lam_min = float(eigvals[0])
            lam_max = float(eigvals[1])
            elongation_ratio = lam_max / max(lam_min, 1e-9)
            if elongation_ratio < elongation_threshold:
                # 近圆形物体, 不重写朝向 (调用方用径向朝下)
                return None, elongation_ratio, None
            # 短轴 = 最小特征值对应的特征向量 (eigh 返回列向量, 升序)
            short_axis_2d = eigvecs[:, 0]  # [vx, vy] in XY 平面
            # 构造夹爪朝向 (base 系): Z=朝下, X=短轴, Y=Z×X
            z_axis = np.array([0.0, 0.0, -1.0])
            x_axis = np.array([short_axis_2d[0], short_axis_2d[1], 0.0])
            x_norm = float(np.linalg.norm(x_axis))
            if x_norm < 1e-6:
                return None, elongation_ratio, None
            x_axis = x_axis / x_norm
            y_axis = np.cross(z_axis, x_axis)
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm < 1e-6:
                return None, elongation_ratio, None
            y_axis = y_axis / y_norm
            # 旋转矩阵 R = [X | Y | Z] (列向量)
            R = np.column_stack([x_axis, y_axis, z_axis])
            R4 = np.eye(4)
            R4[:3, :3] = R
            quat = tf_transformations.quaternion_from_matrix(R4)  # [x, y, z, w]
            return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])], elongation_ratio, centroid_z
        except Exception as e:
            self.get_logger().warning(f"PCA 短轴对齐失败: {e}", throttle_duration_sec=5.0)
            return None, None, None

    def detect_and_publish(self):
        """主检测函数"""
        if not self.rgb_ready or self.latest_rgb is None:
            return
            
        self.frame_count += 1
        display_img = self.latest_rgb.copy()
        try:
            # YOLO 检测
            results = self.model(self.latest_rgb, conf=self.CONF_THRESHOLD, verbose=False)
            
            # ==================== 多目标检测 ====================
            # 遍历 YOLO 输出的所有检测框, 每个都: 绘制标注 + 坐标变换 + 收集到 objects 列表。
            # 主物体 (= 最高置信度的有效物体) 信息填到顶层字段, 保持与 auto_sorting_action.py
            # 的向后兼容 (_two_stage_refine 仍能读 detected / base_position_m)。
            boxes = results[0].boxes
            if len(boxes) > 0:
                self.detection_count += 1
                # 按置信度从高到低排序, 主物体 = boxes_sorted[0]
                boxes_sorted = sorted(
                    boxes,
                    key=lambda b: float(b.conf[0].cpu().numpy()),
                    reverse=True
                )
            else:
                boxes_sorted = []

            # eye-to-hand: 无需选择机械臂位姿, 相机变换是常量
            selected_robot_pose = None
            pose_time_diff = None
            robot_position = None  # eye-to-hand: 不跟踪机械臂位置

            # 不同物体用不同颜色框 (BGR), 循环使用
            box_colors = [
                (0, 255, 0),    # 绿
                (255, 0, 0),    # 蓝
                (0, 165, 255),  # 橙
                (255, 0, 255),  # 品红
                (0, 255, 255),  # 黄
                (255, 255, 0),  # 青
                (128, 0, 128),  # 紫
            ]

            objects_list = []
            primary_info = None  # 主物体完整信息 (最高置信度的有效物体)
            matched_track_ids = set()  # 本帧已匹配的 track_id, 避免重复匹配

            for idx, box in enumerate(boxes_sorted):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id]

                # 中心点
                center_u = int((x1 + x2) / 2)
                center_v = int((y1 + y2) / 2)

                # 真实尺寸 (按类别查询, 用于深度估计和体积)
                real_width, real_height = self._get_class_dimensions(class_name)
                volume = self.estimate_object_volume(real_width, real_height, shape="sphere")

                # ========== 深度估计: 优先 D455 深度相机, 失败时回退到单目 ==========
                # 深度来源选择: 优先 D455, 失败时回退到单目
                depth_source = "monocular"  # 默认单目 (D455失败或未启用时)
                if self.use_depth_camera:
                    self._depth_total_count += 1
                    depth_m = self.depth_from_camera([x1, y1, x2, y2])
                    if depth_m is None:
                        # D455 深度无效 (反光/透明/超出范围), 自动回退到单目
                        depth_m = self.monocular_depth_from_bbox([x1, y1, x2, y2], class_name)
                        self._depth_fallback_count += 1
                        depth_source = "monocular_fallback"  # D455失败→单目(用预设尺寸)
                    else:
                        depth_source = "d455_stereo"  # D455双目深度成功
                else:
                    depth_m = self.monocular_depth_from_bbox([x1, y1, x2, y2], class_name)
                    depth_source = "monocular_only"  # 未启用深度相机

                # 原始坐标变换 (深度成功时计算, 用于喂给滤波器)
                # eye-to-hand: 直接相机→基座 (常量矩阵), 无需 ee_coords
                camera_coords = None
                ee_coords = None
                base_coords = None
                if depth_m is not None:
                    camera_coords = self.monocular_pixel_to_camera_coords(center_u, center_v, depth_m)
                    base_coords, _ = self.transform_camera_to_base(camera_coords)

                # ========== 滤波: 匹配 track + EMA 更新 ==========
                # 用原始观测更新 track, 再用滤波值替换发布值 (画框/坐标都用滤波后)
                raw_obs = {
                    'x1': float(x1), 'y1': float(y1), 'x2': float(x2), 'y2': float(y2),
                    'center_u': center_u, 'center_v': center_v,
                    'depth': depth_m,
                    'camera_coords': camera_coords,
                    'ee_coords': ee_coords,
                    'base_coords': base_coords,
                    'conf': conf,
                }
                track = self._find_matching_track(class_name, center_u, center_v, matched_track_ids)
                if track is None:
                    # 新物体, 创建 track
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    track = ObjectTrack(track_id, class_name, raw_obs, self.frame_count, self.filter_alpha)
                    self._tracks[track_id] = track
                else:
                    matched_track_ids.add(track.track_id)
                    track.update(raw_obs, self.frame_count, self.filter_alpha)

                # 用滤波值替换 (画框/标注/发布都用平滑后的值)
                x1, y1, x2, y2 = track.s_x1, track.s_y1, track.s_x2, track.s_y2
                center_u = int((track.s_x1 + track.s_x2) / 2)
                center_v = int((track.s_y1 + track.s_y2) / 2)
                depth_m = track.s_depth
                camera_coords = track.s_cam
                ee_coords = track.s_ee
                base_coords = track.s_base

                # 距离 (用滤波后的基座坐标重算)
                distance_to_robot = None
                if base_coords is not None and robot_position is not None:
                    distance_to_robot = self.calculate_distance(base_coords, robot_position)

                # ── 绘制检测框 + 类别标签 (用滤波后的 bbox) ──
                box_color = box_colors[idx % len(box_colors)]
                cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                cv2.circle(display_img, (center_u, center_v), 4, (0, 0, 255), -1)
                label = f"#{idx+1} {class_name}: {conf:.2f}"
                cv2.putText(display_img, label, (int(x1), max(int(y1)-8, 12)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

                # 框旁标注位置 (优先放框右下外侧, 出界则放框内左上)
                coord_label_x = int(x2) + 4
                coord_label_y = int(y2)
                if coord_label_x > display_img.shape[1] - 200:
                    coord_label_x = int(x1) + 4
                    coord_label_y = int(y1) + 18

                if depth_m is None:
                    # 深度估计失败 (像素尺寸异常或超出合理范围, 或 track 从未成功估计深度)
                    cv2.putText(display_img, "depth=N/A",
                               (coord_label_x, coord_label_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                    obj_info = {
                        "index": idx + 1,
                        "object_name": class_name,
                        "confidence": conf,
                        "class_id": cls_id,
                        "bbox_pixel": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2)
                        },
                        "size_m": {
                            "width": round(real_width, 4),
                            "height": round(real_height, 4),
                            "diameter": round((real_width + real_height) / 2, 4),
                        },
                        "volume_m3": round(volume, 6),
                        "depth_available": False,
                        "depth_source": "none",  # D455+单目均失败
                        "filtered": True,
                        "track_id": track.track_id,
                    }
                    if self.latest_rgb_stamp is not None:
                        obj_info["rgb_stamp_s"] = round(self.latest_rgb_stamp, 6)
                    objects_list.append(obj_info)
                    continue

                # 深度成功, obj_x/obj_y/obj_z 从滤波后的 camera_coords 取
                obj_x, obj_y, obj_z = camera_coords

                # 框旁标注: 深度 + 基座坐标 (用滤波值)
                cv2.putText(display_img, f"d={depth_m:.2f}m",
                           (coord_label_x, coord_label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                if base_coords is not None:
                    cv2.putText(display_img,
                               f"B({base_coords[0]:.2f},{base_coords[1]:.2f},{base_coords[2]:.2f})",
                               (coord_label_x, coord_label_y + 16),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

                # ── 估计直径 (bbox像素 + D455深度, 不依赖预设尺寸) ──
                # diam = bbox平均像素尺寸 × 深度 / fx (针孔模型反投影)
                _est_diam = ((float(x2 - x1) + float(y2 - y1)) / 2.0) * depth_m / self.fx
                # ── 收集物体信息到 objects 列表 (用滤波值) ──
                obj_info = {
                    "index": idx + 1,
                    "object_name": class_name,
                    "confidence": conf,
                    "class_id": cls_id,
                    "bbox_pixel": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    },
                    "camera_position_m": {
                        "pixel_u": center_u,
                        "pixel_v": center_v,
                        "depth_m": round(obj_z, 4)
                    },
                    "end_effector_position_m": None,  # eye-to-hand: 不再计算 ee_coords
                    "monocular_depth_m": round(depth_m, 4),
                    "depth_source": depth_source,  # d455_stereo | monocular_fallback | monocular_only
                    "estimated_diameter_m": round(_est_diam, 4),  # D455深度估计, 不依赖预设
                    "size_m": {
                        "width": round(real_width, 4),
                        "height": round(real_height, 4),
                        "diameter": round((real_width + real_height) / 2, 4),
                        "note": "known_size, not estimated from depth"
                    },
                    "volume_m3": round(volume, 6),
                    "depth_available": True,
                    "filtered": True,  # 标记本帧坐标已经过 EMA 滤波
                    "track_id": track.track_id,
                }
                if base_coords is not None:
                    obj_info["base_position_m"] = {
                        "x": round(base_coords[0], 4),
                        "y": round(base_coords[1], 4),
                        "z": round(base_coords[2], 4)
                    }
                if distance_to_robot is not None:
                    obj_info["distance_to_robot_m"] = round(distance_to_robot, 4)
                if self.latest_rgb_stamp is not None:
                    obj_info["rgb_stamp_s"] = round(self.latest_rgb_stamp, 6)

                objects_list.append(obj_info)

                # ── 主物体 (第一个有效物体 = 最高置信度):
                #    发布 RViz 可视化 + /object_3d_position (Float64MultiArray, 向后兼容) ──
                if primary_info is None:
                    primary_info = obj_info
                    self.publish_object_visualization(camera_coords, base_coords)
                    # ── PCA 短轴对齐: 用当前帧深度图 + 滤波后 bbox 计算
                    #    夹爪朝向 (顶视 + X 轴对齐物体短轴), 替代已移除的 GraspNet 6DoF ──
                    if base_coords is not None and self.latest_depth is not None:
                        grasp_ori, elong, centroid_z = self._compute_grasp_orientation(
                            [x1, y1, x2, y2], self.latest_depth, float(base_coords[2]))
                        if centroid_z is not None:
                            _surf_z = float(base_coords[2])
                            self.get_logger().info(
                                f"📐 质心z={centroid_z:.4f} 表面z={_surf_z:.4f} "
                                f"Δ={_surf_z-centroid_z:.4f} (若Δ=0.025已限幅; 若表面z偏低调大depth_offset_m)",
                                throttle_duration_sec=2.0)
                    else:
                        grasp_ori, elong, centroid_z = None, None, None
                    primary_info['grasp_orientation'] = grasp_ori
                    primary_info['elongation_ratio'] = elong
                    primary_info['centroid_base_z'] = centroid_z

                    coord_msg = Float64MultiArray()
                    # eye-to-hand: ee_coords 不再计算, 用 0.0 占位 (向后兼容 Float64MultiArray 格式)
                    coord_data = [obj_x, obj_y, obj_z,
                                 0.0, 0.0, 0.0,  # ee_coords 占位 (eye-to-hand 不适用)
                                 real_width, real_height, volume,
                                 conf, float(cls_id), depth_m]
                    if distance_to_robot is not None:
                        coord_data.append(distance_to_robot)
                    coord_msg.data = coord_data
                    self.pub_object_pose.publish(coord_msg)

            # 循环结束: 清理过期 track + 未匹配 track miss_count++
            self._cleanup_stale_tracks(self.frame_count)

            # eye-to-hand: 无需恢复 robot_current_pose (不再使用)

            # 画面左上角汇总: 检测到的物体数量 + 主物体坐标
            cv2.putText(display_img, f"Objects: {len(objects_list)}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if primary_info is not None:
                p = primary_info
                cv2.putText(display_img,
                           f"Primary: {p['object_name']} d={p['monocular_depth_m']:.2f}m",
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1)
                bp = p.get("base_position_m")
                if bp:
                    cv2.putText(display_img,
                               f"Base: ({bp['x']:.3f},{bp['y']:.3f},{bp['z']:.3f})",
                               (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # ── 发布 /detection_info (顶层=主物体字段 + objects 列表, 向后兼容) ──
            info_dict = {
                "detected": primary_info is not None,
                "method": (primary_info.get("depth_source", "unknown") if primary_info is not None else "unknown"),
                "header_stamp": self.latest_rgb_stamp,  # YOLO 检测时刻, 用于 grasp_pose_node 时间同步
                "objects": objects_list,
                "objects_count": len(objects_list),
            }
            if primary_info is not None:
                info_dict["object_name"] = primary_info["object_name"]
                info_dict["confidence"] = primary_info["confidence"]
                info_dict["bbox_pixel"] = primary_info["bbox_pixel"]
                info_dict["camera_position_m"] = primary_info["camera_position_m"]
                info_dict["end_effector_position_m"] = primary_info["end_effector_position_m"]
                info_dict["monocular_depth_m"] = primary_info["monocular_depth_m"]
                info_dict["size_m"] = primary_info["size_m"]
                info_dict["volume_m3"] = primary_info["volume_m3"]
                if "base_position_m" in primary_info:
                    info_dict["base_position_m"] = primary_info["base_position_m"]
                if "distance_to_robot_m" in primary_info:
                    info_dict["distance_to_robot_m"] = primary_info["distance_to_robot_m"]
                # ── PCA 短轴对齐朝向 (替代 GraspNet) ──
                if "grasp_orientation" in primary_info:
                    info_dict["grasp_orientation"] = primary_info["grasp_orientation"]
                    info_dict["elongation_ratio"] = primary_info.get("elongation_ratio")
                # ── 质心 z (D455 深度点云估算, 不依赖预设尺寸, 供抓取高度补偿) ──
                if "centroid_base_z" in primary_info and primary_info["centroid_base_z"] is not None:
                    info_dict["centroid_base_z"] = primary_info["centroid_base_z"]
                # ── 估计直径 (bbox+D455深度, 不依赖预设尺寸, 供夹爪闭合目标) ──
                if "estimated_diameter_m" in primary_info:
                    info_dict["estimated_diameter_m"] = primary_info["estimated_diameter_m"]
                # ── 随动抓取接口占位 (中期实现: 光流/帧间位移估算物体速度) ──
                info_dict["velocity_mps"] = None

            # eye-to-hand: 不再发布 used_robot_pose / tcp_pose_m (相机变换为常量)
            if self.latest_rgb_stamp is not None:
                info_dict["rgb_stamp_s"] = round(self.latest_rgb_stamp, 6)

            info_msg = String()
            info_msg.data = json.dumps(info_dict, ensure_ascii=False)
            self.pub_detection_info.publish(info_msg)

            # 终端日志 (多目标简要)
            if objects_list:
                self.get_logger().info("=" * 60)
                self.get_logger().info(f"🎯 检测到 {len(objects_list)} 个物体:")
                for obj in objects_list:
                    bp = obj.get("base_position_m", {})
                    bp_str = f" 基座({bp['x']:.3f},{bp['y']:.3f},{bp['z']:.3f})" if bp else ""
                    self.get_logger().info(
                        f"  #{obj['index']} {obj['object_name']} conf={obj['confidence']:.3f} "
                        f"depth={obj['monocular_depth_m']:.3f}m{bp_str}"
                    )
                self.get_logger().info("=" * 60)
        except Exception as e:
            self.get_logger().error(f"检测错误: {e}")
            cv2.putText(display_img, f"Error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # ── 在画面上叠加曝光状态 (无论 standalone 还是 GUI 模式都显示) ──
        # 放在画面底部避免与检测信息 (y_offset 从 30 开始) 重叠
        exposure_text = f"曝光: {'自动' if self.auto_exposure else int(self.current_exposure)}"
        cv2.putText(display_img, exposure_text, (10, display_img.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ── 每帧都发布标注后的 JPEG，供 GUI 或外部订阅 ──
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

        # ── standalone 模式：OpenCV 窗口 + 键盘曝光调节 ──
        if self.show_gui_window:
            cv2.imshow("Object Detection", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.get_logger().info("用户退出")
                if rclpy.ok():
                    rclpy.shutdown()
            elif key == ord('+') or key == ord('='):
                # 手动模式下增加曝光
                if not self.auto_exposure:
                    step = 100 if self.current_exposure >= 100 else 10
                    self.set_exposure(False, self.current_exposure + step)
            elif key == ord('-') or key == ord('_'):
                # 手动模式下减少曝光
                if not self.auto_exposure:
                    step = 100 if self.current_exposure >= 100 else 10
                    self.set_exposure(False, self.current_exposure - step)
            elif key == ord('a'):
                # 切换自动/手动曝光
                self.set_exposure(not self.auto_exposure)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    
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