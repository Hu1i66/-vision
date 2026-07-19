#!/usr/bin/env python3
from collections import deque
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64MultiArray, String
import cv2
import numpy as np
import json
import math
import tf_transformations

class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")
        
        # ================== 模型配置 ==================
        self.MODEL_PATH = "/home/lxf/orange_dataset/runs/detect/runs/detect/orange_exp4/weights/best.pt"
        self.CONF_THRESHOLD = 0.1
        # 默认改为使用 YOLO 检测（可通过 ROS 参数 detection_mode 覆盖）
        self.detection_mode = self.declare_parameter("detection_mode", "yolo").value.lower()
        self.marker_pose_topic = self.declare_parameter("marker_pose_topic", "/aruco_single/pose").value
        self.marker_image_topic = self.declare_parameter("marker_image_topic", "/aruco_single/result").value
        self.marker_id = int(self.declare_parameter("marker_id", 582).value)
        self.marker_size_m = float(self.declare_parameter("marker_size_m", 0.1).value)
        
        # ================== 相机参数 ==================
        self.fx = 378.394659861614
        self.fy = 379.366916262423
        self.cx = 330.140969430714
        self.cy = 246.095530649072
        self.skew = 1.25667775035477
        self.depth_scale = 0.001

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
        
        # ================== 手眼标定数据（相机到机械臂末端）==================
        # 平移向量 (单位: 米)
        self.camera_to_end_effector_translation = np.array([
            -0.061010852967003565,  # X
            -0.008094796521449572,  # Y
            0.029089713469040795    # Z
        ])

        # 可选覆盖：允许在运行时通过参数微调手眼平移偏置（JSON: {"dx":.., "dy":.., "dz":..}）
        self.translation_override_str = self.declare_parameter(
            "camera_to_end_effector_translation_override",
            ""
        ).value
        try:
            if self.translation_override_str:
                ov = json.loads(self.translation_override_str)
                dx = float(ov.get('dx', 0.0))
                dy = float(ov.get('dy', 0.0))
                dz = float(ov.get('dz', 0.0))
                self.get_logger().info(f"应用手眼平移覆盖: dx={dx}, dy={dy}, dz={dz}")
                self.camera_to_end_effector_translation += np.array([dx, dy, dz])
        except Exception as e:
            self.get_logger().warning(f"解析 camera_to_end_effector_translation_override 失败: {e}")
        
        # 四元数 (x, y, z, w)
        self.camera_to_end_effector_quat = np.array([
            -0.12871193615712131,   # qx
            0.1305607862106386,     # qy
            -0.7125529515814637,    # qz
            0.6772410278381616      # qw
        ])
        
        # RPY 角度 (弧度)
        self.camera_to_end_effector_rpy = np.array([
            -0.3687060786240621,    # roll
            -0.006585945538182644,  # pitch
            -1.620373363544811      # yaw
        ])
        
        # ================== 构建完整的变换矩阵 ==================
        # 方法1：使用四元数构建旋转矩阵
        self.rotation_matrix_cam_to_ee = self.quaternion_to_rotation_matrix(
            self.camera_to_end_effector_quat
        )
        
        # 方法2：使用 RPY 角度构建旋转矩阵（验证用）
        self.rotation_matrix_cam_to_ee_rpy = self.rpy_to_rotation_matrix(
            self.camera_to_end_effector_rpy
        )
        
        # 构建 4x4 齐次变换矩阵 (相机坐标系 -> 机械臂末端坐标系)
        self.transform_cam_to_ee = np.eye(4)
        self.transform_cam_to_ee[:3, :3] = self.rotation_matrix_cam_to_ee
        self.transform_cam_to_ee[:3, 3] = self.camera_to_end_effector_translation
        
        # 计算机械臂末端到相机的逆变换（用于验证）
        self.transform_ee_to_cam = np.linalg.inv(self.transform_cam_to_ee)
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("🔧 手眼标定参数已加载:")
        self.get_logger().info(f"   平移: X={self.camera_to_end_effector_translation[0]:.6f}m")
        self.get_logger().info(f"         Y={self.camera_to_end_effector_translation[1]:.6f}m")
        self.get_logger().info(f"         Z={self.camera_to_end_effector_translation[2]:.6f}m")
        self.get_logger().info(f"   四元数: qx={self.camera_to_end_effector_quat[0]:.6f}")
        self.get_logger().info(f"           qy={self.camera_to_end_effector_quat[1]:.6f}")
        self.get_logger().info(f"           qz={self.camera_to_end_effector_quat[2]:.6f}")
        self.get_logger().info(f"           qw={self.camera_to_end_effector_quat[3]:.6f}")
        self.get_logger().info(f"   RPY: roll={math.degrees(self.camera_to_end_effector_rpy[0]):.2f}°")
        self.get_logger().info(f"         pitch={math.degrees(self.camera_to_end_effector_rpy[1]):.2f}°")
        self.get_logger().info(f"         yaw={math.degrees(self.camera_to_end_effector_rpy[2]):.2f}°")
        self.get_logger().info("=" * 60)
        
        # ================== 机械臂当前位姿（如果连接了机械臂）==================
        self.robot_current_pose = None
        self.robot_pose_topic = self.declare_parameter(
            "robot_pose_topic",
            "/feedback/tcp_pose"
        ).value
        self.latest_marker_pose = None
        self.latest_marker_stamp = None
        self.latest_marker_image = None
        self.latest_marker_image_stamp = None
        # 可选：提供物体在机械臂基坐标系下的“真实”位置，用于手眼标定/调试
        # 格式: JSON 字符串，例如 '{"x":0.2, "y":0.0, "z":0.01}'
        self.ground_truth_base_str = self.declare_parameter(
            "ground_truth_base",
            ""
        ).value
        try:
            self.ground_truth_base = json.loads(self.ground_truth_base_str) if self.ground_truth_base_str else None
        except Exception:
            self.get_logger().warning("ground_truth_base 参数解析失败，忽略")
            self.ground_truth_base = None
        
        # ================== 初始化 ==================
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_rgb = None
        self.latest_rgb_stamp = None
        self.rgb_ready = False
        self.depth_ready = False
        self.frame_count = 0
        
        # 统计信息
        self.detection_count = 0
        self.depth_fail_count = 0
        self.status_printed = False
        
        # ================== 创建发布/订阅 ==================
        self.pub_object_pose = self.create_publisher(Float64MultiArray, "/object_3d_position", 10)
        self.pub_detection_info = self.create_publisher(String, "/detection_info", 10)

        # 保留一段机械臂位姿历史，按图像时间戳取最近值，避免图像/位姿错位
        self.robot_pose_history = deque(maxlen=500)
        self.robot_pose_sync_tolerance = 0.15
        
        # 订阅机械臂位姿（默认使用当前系统里真实存在的 TCP 位姿话题）
        self.sub_robot_pose = self.create_subscription(
            PoseStamped,
            self.robot_pose_topic,
            self.robot_pose_cb,
            10
        )

        if self.detection_mode == "aruco":
            self.sub_marker_pose = self.create_subscription(
                PoseStamped,
                self.marker_pose_topic,
                self.marker_pose_cb,
                10
            )
            self.sub_marker_image = self.create_subscription(
                Image,
                self.marker_image_topic,
                self.marker_image_cb,
                10
            )
            self.get_logger().info(f"使用 ArUco 标定板模式，订阅 {self.marker_pose_topic}")
            self.get_logger().info(f"ArUco 图像话题: {self.marker_image_topic}")
        else:
            # 订阅相机话题
            self.sub_rgb = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_cb, 10)
            self.sub_depth = self.create_subscription(Image, "/camera/camera/depth/image_rect_raw", self.depth_cb, 10)
            
            self.get_logger().info("正在加载 YOLO 模型...")
            from ultralytics import YOLO
            self.model = YOLO(self.MODEL_PATH)
            self.get_logger().info(f"✅ 模型加载成功，类别: {self.model.names}")
        
        # 定时器
        self.create_timer(0.1, self.detect_and_publish)
        self.status_timer = self.create_timer(5.0, self.print_status)
        
        # OpenCV 窗口
        cv2.namedWindow("Object Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Object Detection", 960, 540)
        
        self.get_logger().info("✅ 节点启动，等待相机数据...")
        self.get_logger().info(f"🔍 检测模式: {self.detection_mode}")
        self.get_logger().info(f"🔍 检测阈值: {self.CONF_THRESHOLD}")
        self.get_logger().info(f"🔳 标定板边长: {self.marker_size_m:.4f}m")
        self.get_logger().info(f"🔳 ArUco marker_id: {self.marker_id}")
        self.get_logger().info(f"🤖 机械臂位姿话题: {self.robot_pose_topic}")
        
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
    
    def robot_pose_cb(self, msg):
        """接收机械臂当前位姿"""
        self.robot_current_pose = msg
        stamp = msg.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.robot_pose_history.append((stamp_sec, msg))

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
    def marker_pose_cb(self, msg):
        """接收 ArUco 标定板位姿"""
        self.latest_marker_pose = msg
        self.latest_marker_stamp = self._stamp_to_sec(msg.header.stamp)

    def marker_image_cb(self, msg):
        """接收 ArUco 标定板可视化图像"""
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            self.latest_marker_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            self.latest_marker_image_stamp = self._stamp_to_sec(msg.header.stamp)
        except Exception as e:
            self.get_logger().error(f"ArUco 图像转换错误: {e}")

    def _stamp_to_sec(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _get_best_robot_pose(self, target_stamp_sec):
        if not self.robot_pose_history:
            return self.robot_current_pose, None

        best_stamp, best_msg = min(
            self.robot_pose_history,
            key=lambda item: abs(item[0] - target_stamp_sec)
        )
        return best_msg, abs(best_stamp - target_stamp_sec)
        
    def print_status(self):
        """打印状态信息"""
        if not self.status_printed:
            self.status_printed = True
            self.get_logger().info("=" * 60)
            self.get_logger().info("📊 状态统计:")
            if self.detection_mode == "aruco":
                self.get_logger().info(f"  - ArUco 位姿: {'就绪' if self.latest_marker_pose else '等待中'}")
            else:
                self.get_logger().info(f"  - RGB 数据: {'就绪' if self.rgb_ready else '等待中'}")
                self.get_logger().info(f"  - 深度数据: {'就绪' if self.depth_ready else '等待中'}")
            self.get_logger().info(f"  - 机械臂位姿: {'就绪' if self.robot_current_pose else '未连接'}")
            self.get_logger().info(f"  - 已处理帧数: {self.frame_count}")
            self.get_logger().info(f"  - 检测到物体: {self.detection_count} 次")
            self.get_logger().info("=" * 60)
            self.status_timer.cancel()
        
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
        """接收深度图"""
        try:
            depth_img = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            self.latest_depth = depth_img
            self.latest_depth_stamp = self._stamp_to_sec(msg.header.stamp)
            
            if not self.depth_ready:
                self.depth_ready = True
                self.get_logger().info(f"✅ 深度图就绪: {msg.width}x{msg.height}")
        except Exception as e:
            self.get_logger().error(f"深度图转换错误: {e}")
            
    def get_depth_at_pixel(self, x, y, window_size=5):
        """获取像素点的深度值（米）"""
        if self.latest_depth is None:
            return None
            
        h, w = self.latest_depth.shape
        if x < 0 or x >= w or y < 0 or y >= h:
            return None
            
        half = window_size // 2
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(w, x + half + 1)
        y2 = min(h, y + half + 1)
        
        depth_window = self.latest_depth[y1:y2, x1:x2]
        valid_depths = depth_window[depth_window > 0]
        
        if len(valid_depths) == 0:
            return None
            
        depth_mm = np.median(valid_depths)
        depth_m = depth_mm * self.depth_scale
        
        if depth_m < 0.1 or depth_m > 5.0:
            return None
            
        return depth_m
        
    def pixel_to_camera_coords(self, u, v):
        """像素坐标转相机坐标系3D坐标"""
        depth = self.get_depth_at_pixel(u, v)
        if depth is None:
            return None
            
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        
        return np.array([x, y, z])

    def get_depth_in_box(self, box):
        """获取检测框内的代表性深度（米）"""
        if self.latest_depth is None:
            return None

        x1, y1, x2, y2 = [int(v) for v in box]
        h, w = self.latest_depth.shape
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        depth_window = self.latest_depth[y1:y2, x1:x2]
        valid_depths = depth_window[depth_window > 0]
        if len(valid_depths) == 0:
            return None

        depth_mm = np.median(valid_depths)
        depth_m = depth_mm * self.depth_scale
        if depth_m < 0.1 or depth_m > 5.0:
            return None

        return depth_m
    
    def transform_camera_to_end_effector(self, camera_coords):
        """
        将相机坐标系下的坐标转换到机械臂末端坐标系
        使用完整的手眼标定变换矩阵（包含旋转）
        """
        # 转换为齐次坐标
        point_homogeneous = np.append(camera_coords, 1)
        
        # 应用变换
        end_effector_coords = self.transform_cam_to_ee @ point_homogeneous
        
        return end_effector_coords[:3]
    
    def transform_end_effector_to_base(self, end_effector_coords):
        """
        将机械臂末端坐标系下的坐标转换到基坐标系
        需要机械臂当前位姿
        """
        if self.robot_current_pose is None:
            return None
        
        # 获取机械臂末端的位姿
        ee_x = self.robot_current_pose.pose.position.x
        ee_y = self.robot_current_pose.pose.position.y
        ee_z = self.robot_current_pose.pose.position.z
        
        ee_qx = self.robot_current_pose.pose.orientation.x
        ee_qy = self.robot_current_pose.pose.orientation.y
        ee_qz = self.robot_current_pose.pose.orientation.z
        ee_qw = self.robot_current_pose.pose.orientation.w
        
        # 四元数转旋转矩阵
        rotation_matrix = tf_transformations.quaternion_matrix([ee_qx, ee_qy, ee_qz, ee_qw])
        R_ee_to_base = rotation_matrix[:3, :3]
        
        # 机械臂末端位置
        T_ee_to_base = np.array([ee_x, ee_y, ee_z])
        
        # 转换
        base_coords = R_ee_to_base @ end_effector_coords + T_ee_to_base
        
        return base_coords
    
    def transform_camera_to_base(self, camera_coords):
        """
        完整的转换：相机坐标系 -> 机械臂末端坐标系 -> 机械臂基坐标系
        """
        # 步骤1：相机坐标系 -> 机械臂末端坐标系
        ee_coords = self.transform_camera_to_end_effector(camera_coords)
        
        # 步骤2：机械臂末端坐标系 -> 机械臂基坐标系
        base_coords = self.transform_end_effector_to_base(ee_coords)
        
        return base_coords, ee_coords
    
    def get_robot_current_position(self):
        """
        获取机械臂当前位置（基坐标系下）
        """
        if self.robot_current_pose is not None:
            return np.array([
                self.robot_current_pose.pose.position.x,
                self.robot_current_pose.pose.position.y,
                self.robot_current_pose.pose.position.z
            ])
        else:
            return None
    
    def calculate_distance(self, point1, point2):
        """计算两点之间的欧氏距离"""
        if point1 is None or point2 is None:
            return None
        return np.linalg.norm(point1 - point2)
    
    def estimate_object_size(self, box, depth):
        """估计物体的实际尺寸"""
        x1, y1, x2, y2 = box
        pixel_width = x2 - x1
        pixel_height = y2 - y1
        
        real_width = pixel_width * depth / self.fx
        real_height = pixel_height * depth / self.fy
        
        return real_width, real_height
    
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

    def detect_and_publish(self):
        """主检测函数"""
        if self.detection_mode == "aruco":
            if self.latest_marker_pose is None:
                blank = np.zeros((540, 960, 3), dtype=np.uint8)
                cv2.putText(blank, "WAITING FOR ARUCO POSE...", (40, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.putText(blank, f"marker_id={self.marker_id}, marker_size={self.marker_size_m:.3f}m", (40, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow("Object Detection", blank)
                cv2.waitKey(1)
                return

            self.frame_count += 1
            camera_coords = np.array([
                self.latest_marker_pose.pose.position.x,
                self.latest_marker_pose.pose.position.y,
                self.latest_marker_pose.pose.position.z,
            ])
            obj_x, obj_y, obj_z = camera_coords

            selected_robot_pose = self.robot_current_pose
            pose_time_diff = None
            if self.latest_marker_stamp is not None:
                selected_robot_pose, pose_time_diff = self._get_best_robot_pose(self.latest_marker_stamp)
            if selected_robot_pose is not None and pose_time_diff is not None and pose_time_diff > self.robot_pose_sync_tolerance:
                self.get_logger().warning(
                    f"标定板与机械臂位姿时间差较大: {pose_time_diff:.3f}s, 结果可能抖动"
                )

            original_robot_pose = self.robot_current_pose
            self.robot_current_pose = selected_robot_pose
            ee_coords = self.transform_camera_to_end_effector(camera_coords)
            base_coords = None
            if self.robot_current_pose is not None:
                base_coords, _ = self.transform_camera_to_base(camera_coords)
            self.robot_current_pose = original_robot_pose

            robot_position = self.get_robot_current_position()
            distance_to_robot = None
            if base_coords is not None and robot_position is not None:
                distance_to_robot = self.calculate_distance(base_coords, robot_position)

            marker_side_length = self.marker_size_m
            marker_area = marker_side_length * marker_side_length

            coord_msg = Float64MultiArray()
            coord_data = [obj_x, obj_y, obj_z,
                          ee_coords[0], ee_coords[1], ee_coords[2],
                          marker_side_length, marker_area, float(self.marker_id)]
            if base_coords is not None:
                coord_data.extend([base_coords[0], base_coords[1], base_coords[2]])
            if distance_to_robot is not None:
                coord_data.append(distance_to_robot)
            coord_msg.data = coord_data
            self.pub_object_pose.publish(coord_msg)

            info_dict = {
                "detected": True,
                "detector": "aruco_ros",
                "marker_id": self.marker_id,
                "marker_side_length_m": round(marker_side_length, 4),
                "marker_area_m2": round(marker_area, 6),
                "camera_position_m": {
                    "x": round(obj_x, 4),
                    "y": round(obj_y, 4),
                    "z": round(obj_z, 4)
                },
                "end_effector_position_m": {
                    "x": round(ee_coords[0], 4),
                    "y": round(ee_coords[1], 4),
                    "z": round(ee_coords[2], 4)
                },
                "depth_m": round(obj_z, 4)
            }
            if base_coords is not None:
                info_dict["base_position_m"] = {
                    "x": round(base_coords[0], 4),
                    "y": round(base_coords[1], 4),
                    "z": round(base_coords[2], 4)
                }
            if distance_to_robot is not None:
                info_dict["distance_to_robot_m"] = round(distance_to_robot, 4)
            if self.latest_marker_stamp is not None:
                info_dict["marker_stamp_s"] = round(self.latest_marker_stamp, 6)
            if pose_time_diff is not None:
                info_dict["robot_pose_time_diff_s"] = round(pose_time_diff, 6)

            info_msg = String()
            info_msg.data = json.dumps(info_dict, indent=2)
            self.pub_detection_info.publish(info_msg)

            self.get_logger().info("=" * 60)
            self.get_logger().info(f"🎯 ArUco marker_{self.marker_id}: 已检测")
            self.get_logger().info(f"📍 相机坐标: X={obj_x:.4f}m, Y={obj_y:.4f}m, Z={obj_z:.4f}m")
            self.get_logger().info(f"🔧 末端坐标: X={ee_coords[0]:.4f}m, Y={ee_coords[1]:.4f}m, Z={ee_coords[2]:.4f}m")
            self.get_logger().info(f"📐 标定板边长: {marker_side_length:.4f}m")
            if base_coords is not None:
                self.get_logger().info(f"🤖 基座坐标: X={base_coords[0]:.4f}m, Y={base_coords[1]:.4f}m, Z={base_coords[2]:.4f}m")
            if distance_to_robot is not None:
                self.get_logger().info(f"📏 机械臂距离: {distance_to_robot:.4f}m")
            self.get_logger().info("=" * 60)

            display_img = self.latest_marker_image.copy() if self.latest_marker_image is not None else np.zeros((540, 960, 3), dtype=np.uint8)
            y_offset = 40
            cv2.putText(display_img, f"ArUco marker_{self.marker_id}", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y_offset += 35
            cv2.putText(display_img, f"Cam XYZ: ({obj_x:.4f}, {obj_y:.4f}, {obj_z:.4f}) m", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            y_offset += 30
            cv2.putText(display_img, f"EE XYZ: ({ee_coords[0]:.4f}, {ee_coords[1]:.4f}, {ee_coords[2]:.4f}) m", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30
            cv2.putText(display_img, f"Edge: {marker_side_length:.4f} m", (20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if base_coords is not None:
                y_offset += 30
                cv2.putText(display_img, f"Base XYZ: ({base_coords[0]:.4f}, {base_coords[1]:.4f}, {base_coords[2]:.4f}) m", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            if pose_time_diff is not None:
                y_offset += 30
                cv2.putText(display_img, f"Pose dt: {pose_time_diff:.3f}s", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Object Detection", display_img)
            cv2.waitKey(1)
            return

        if not self.rgb_ready or self.latest_rgb is None:
            return
            
        self.frame_count += 1
        display_img = self.latest_rgb.copy()
        
        if not self.depth_ready:
            cv2.putText(display_img, "WAITING FOR DEPTH DATA...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Object Detection", display_img)
            cv2.waitKey(1)
            return
        
        try:
            # YOLO 检测
            results = self.model(self.latest_rgb, conf=self.CONF_THRESHOLD, verbose=False)
            
            if len(results[0].boxes) > 0:
                self.detection_count += 1

                # 获取第一个检测结果
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[cls_id]
                
                # 中心点
                center_u = int((x1 + x2) / 2)
                center_v = int((y1 + y2) / 2)
                
                # 获取相机坐标系下的3D坐标
                camera_coords = self.pixel_to_camera_coords(center_u, center_v)
                
                if camera_coords is not None:
                    obj_x, obj_y, obj_z = camera_coords

                    # 选择与当前图像时间戳最接近的机械臂位姿，减少运动时的坐标跳变
                    selected_robot_pose = self.robot_current_pose
                    pose_time_diff = None
                    if self.latest_rgb_stamp is not None:
                        selected_robot_pose, pose_time_diff = self._get_best_robot_pose(self.latest_rgb_stamp)
                    if selected_robot_pose is not None and pose_time_diff is not None and pose_time_diff > self.robot_pose_sync_tolerance:
                        self.get_logger().warning(
                            f"图像与机械臂位姿时间差较大: {pose_time_diff:.3f}s, 结果可能抖动"
                        )

                    original_robot_pose = self.robot_current_pose
                    self.robot_current_pose = selected_robot_pose
                    
                    # 转换到机械臂末端坐标系
                    ee_coords = self.transform_camera_to_end_effector(camera_coords)
                    
                    # 转换到机械臂基坐标系（如果有机械臂位姿）
                    base_coords = None
                    if self.robot_current_pose is not None:
                        base_coords, _ = self.transform_camera_to_base(camera_coords)

                    self.robot_current_pose = original_robot_pose
                    
                    # 获取机械臂当前位置
                    robot_position = self.get_robot_current_position()
                    
                    # 计算距离（如果有机械臂位姿）
                    distance_to_robot = None
                    if base_coords is not None and robot_position is not None:
                        distance_to_robot = self.calculate_distance(base_coords, robot_position)
                    
                    # 估计物体大小
                    size_depth = self.get_depth_in_box([x1, y1, x2, y2]) or obj_z
                    real_width, real_height = self.estimate_object_size([x1, y1, x2, y2], size_depth)
                    volume = self.estimate_object_volume(real_width, real_height, shape="sphere")
                    
                    # 绘制检测框
                    cv2.rectangle(display_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(display_img, (center_u, center_v), 5, (0, 0, 255), -1)
                    
                    # 显示类别和置信度
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(display_img, label, (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 在图像上显示信息
                    y_offset = 30
                    cv2.putText(display_img, f"Camera XYZ: ({obj_x:.4f}, {obj_y:.4f}, {obj_z:.4f})m", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    y_offset += 25
                    cv2.putText(display_img, f"End-Effector XYZ: ({ee_coords[0]:.4f}, {ee_coords[1]:.4f}, {ee_coords[2]:.4f})m", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    if base_coords is not None:
                        y_offset += 25
                        cv2.putText(display_img, f"Base XYZ: ({base_coords[0]:.4f}, {base_coords[1]:.4f}, {base_coords[2]:.4f})m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    if distance_to_robot is not None:
                        y_offset += 25
                        cv2.putText(display_img, f"Distance to Robot: {distance_to_robot:.4f}m", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    y_offset += 25
                    cv2.putText(display_img, f"Size: W={real_width:.4f}m H={real_height:.4f}m", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    y_offset += 25
                    cv2.putText(display_img, f"Volume: {volume:.6f}m^3", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                    
                    # 发布坐标和尺寸信息
                    coord_msg = Float64MultiArray()
                    coord_data = [obj_x, obj_y, obj_z, 
                                 ee_coords[0], ee_coords[1], ee_coords[2],
                                 real_width, real_height, volume, 
                                 conf, float(cls_id)]
                    if distance_to_robot is not None:
                        coord_data.append(distance_to_robot)
                    coord_msg.data = coord_data
                    self.pub_object_pose.publish(coord_msg)
                    
                    # 发布详细信息
                    info_dict = {
                        "detected": True,
                        "object_name": class_name,
                        "confidence": conf,
                        "camera_position_m": {
                            "x": round(obj_x, 4),
                            "y": round(obj_y, 4),
                            "z": round(obj_z, 4)
                        },
                        "end_effector_position_m": {
                            "x": round(ee_coords[0], 4),
                            "y": round(ee_coords[1], 4),
                            "z": round(ee_coords[2], 4)
                        },
                        "size_m": {
                            "width": round(real_width, 4),
                            "height": round(real_height, 4),
                            "diameter": round((real_width + real_height) / 2, 4)
                        },
                        "volume_m3": round(volume, 6),
                        "depth_m": round(obj_z, 4)
                    }

                    if selected_robot_pose is not None:
                        info_dict["used_robot_pose"] = self.pose_stamped_to_dict(selected_robot_pose)
                        info_dict["tcp_pose_m"] = self.pose_stamped_to_dict(selected_robot_pose)

                    if self.latest_rgb_stamp is not None:
                        info_dict["rgb_stamp_s"] = round(self.latest_rgb_stamp, 6)
                    if self.latest_depth_stamp is not None:
                        info_dict["depth_stamp_s"] = round(self.latest_depth_stamp, 6)
                    if pose_time_diff is not None:
                        info_dict["robot_pose_time_diff_s"] = round(pose_time_diff, 6)
                    
                    if base_coords is not None:
                        info_dict["base_position_m"] = {
                            "x": round(base_coords[0], 4),
                            "y": round(base_coords[1], 4),
                            "z": round(base_coords[2], 4)
                        }
                    
                    if distance_to_robot is not None:
                        info_dict["distance_to_robot_m"] = round(distance_to_robot, 4)
                    
                    info_msg = String()
                    info_msg.data = json.dumps(info_dict, indent=2)
                    self.pub_detection_info.publish(info_msg)
                    
                    # 打印到终端
                    self.get_logger().info("=" * 60)
                    self.get_logger().info(f"🎯 {class_name}: 置信度={conf:.4f}")
                    self.get_logger().info(f"📍 相机坐标: X={obj_x:.4f}m, Y={obj_y:.4f}m, Z={obj_z:.4f}m")
                    self.get_logger().info(f"🔧 末端坐标: X={ee_coords[0]:.4f}m, Y={ee_coords[1]:.4f}m, Z={ee_coords[2]:.4f}m")
                    if selected_robot_pose is not None:
                        self.get_logger().info(
                            f"🦾 TCP原始位姿: X={selected_robot_pose.pose.position.x:.4f}m, "
                            f"Y={selected_robot_pose.pose.position.y:.4f}m, Z={selected_robot_pose.pose.position.z:.4f}m"
                        )
                        self.get_logger().info(
                            f"🧭 TCP原始四元数: x={selected_robot_pose.pose.orientation.x:.6f}, "
                            f"y={selected_robot_pose.pose.orientation.y:.6f}, "
                            f"z={selected_robot_pose.pose.orientation.z:.6f}, "
                            f"w={selected_robot_pose.pose.orientation.w:.6f}"
                        )
                    # 如果提供了地面真实基坐标，计算预期末端坐标并打印误差，辅助诊断
                    if self.ground_truth_base is not None and self.robot_current_pose is not None:
                        try:
                            # 构建机械臂末端到基座的变换 T (基座 <- 末端)
                            ee_qx = self.robot_current_pose.pose.orientation.x
                            ee_qy = self.robot_current_pose.pose.orientation.y
                            ee_qz = self.robot_current_pose.pose.orientation.z
                            ee_qw = self.robot_current_pose.pose.orientation.w
                            ee_px = self.robot_current_pose.pose.position.x
                            ee_py = self.robot_current_pose.pose.position.y
                            ee_pz = self.robot_current_pose.pose.position.z

                            T_ee_to_base = tf_transformations.quaternion_matrix([ee_qx, ee_qy, ee_qz, ee_qw])
                            T_ee_to_base[:3, 3] = np.array([ee_px, ee_py, ee_pz])
                            T_base_to_ee = np.linalg.inv(T_ee_to_base)

                            gt = self.ground_truth_base
                            obj_base = np.array([gt['x'], gt['y'], gt['z'], 1.0])
                            # 预期末端坐标（由基坐标反推到末端坐标系）
                            expected_ee = (T_base_to_ee @ obj_base)[:3]

                            # 由测得末端坐标反推出的基坐标
                            measured_base = (T_ee_to_base @ np.append(ee_coords, 1))[:3]

                            diff_ee = expected_ee - ee_coords
                            diff_base = measured_base - np.array([gt['x'], gt['y'], gt['z']])

                            self.get_logger().info(f"[DEBUG] 预期末端坐标(基->末端反推): {expected_ee}")
                            self.get_logger().info(f"[DEBUG] 末端坐标差(预期 - 测量): {diff_ee}")
                            self.get_logger().info(f"[DEBUG] 由测得末端反推基坐标: {measured_base}")
                            self.get_logger().info(f"[DEBUG] 基坐标差(测量 - 真实): {diff_base}")
                        except Exception as e:
                            self.get_logger().warning(f"DEBUG 计算失败: {e}")
                    
                    if base_coords is not None:
                        self.get_logger().info(f"🤖 基座坐标: X={base_coords[0]:.4f}m, Y={base_coords[1]:.4f}m, Z={base_coords[2]:.4f}m")
                    
                    if distance_to_robot is not None:
                        self.get_logger().info(f"📏 机械臂距离: {distance_to_robot:.4f}m")
                    
                    self.get_logger().info(f"📐 物体尺寸: 宽={real_width:.4f}m, 高={real_height:.4f}m")
                    self.get_logger().info(f"📦 估计体积: {volume:.6f}m^3")
                    self.get_logger().info("=" * 60)
                    
                else:
                    self.depth_fail_count += 1
        except Exception as e:
            self.get_logger().error(f"检测错误: {e}")
            cv2.putText(display_img, f"Error: {str(e)[:50]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.imshow("Object Detection", display_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            self.get_logger().info("用户退出")
            if rclpy.ok():
                rclpy.shutdown()

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