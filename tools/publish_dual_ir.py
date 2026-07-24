#!/usr/bin/env python3
"""
D455 双红外流 ROS2 发布器
===========================
绕过 realsense2_camera 限制，使用 pyrealsense2 直接读取左/右红外图像，
发布为 ROS2 sensor_msgs/Image 话题，供 cameracalibrator 双目标定使用。

发布话题:
  /camera/camera/infra1/image_rect_raw  — 左红外
  /camera/camera/infra2/image_rect_raw  — 右红外

用法:
  source /home/lxf/orange_dataset/.venv/bin/activate
  python3 tools/publish_dual_ir.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np


class DualIRPublisher(Node):
    def __init__(self):
        super().__init__("dual_ir_publisher")

        self.bridge = CvBridge()

        # 发布话题
        self.pub_infra1 = self.create_publisher(Image, "/camera/camera/infra1/image_rect_raw", 10)
        self.pub_infra2 = self.create_publisher(Image, "/camera/camera/infra2/image_rect_raw", 10)

        # ── 初始化 RealSense pipeline ──
        self.pipe = rs.pipeline()
        config = rs.config()
        # D455 红外相机依赖深度模块，必须同时开启深度流
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)  # 左红外
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 15)  # 右红外

        self.get_logger().info("启动 RealSense 双红外流...")
        self.profile = self.pipe.start(config)

        # 获取并打印内参
        for idx in (1, 2):
            s = self.profile.get_stream(rs.stream.infrared, idx).as_video_stream_profile()
            intr = s.get_intrinsics()
            self.get_logger().info(f"  Infra{idx}: {intr.width}x{intr.height} "
                                  f"fx={intr.fx:.2f} fy={intr.fy:.2f} "
                                  f"cx={intr.cx:.2f} cy={intr.cy:.2f} "
                                  f"model={intr.model}")

        # 禁用红外投影器（标定不需要散斑图案干扰棋盘格检测）
        try:
            sensor = self.profile.get_device().query_sensors()[0]
            sensor.set_option(rs.option.emitter_enabled, 0)
            self.get_logger().info("红外投影器已关闭 (emitter_enabled=0)")
        except Exception as e:
            self.get_logger().warn(f"无法设置 emitter_enabled: {e}")

        self.get_logger().info("✅ 双红外流已上线，发布话题:")
        self.get_logger().info("   左红外: /camera/camera/infra1/image_rect_raw")
        self.get_logger().info("   右红外: /camera/camera/infra2/image_rect_raw")

        # 定时器: 15Hz 读取并发布
        self.create_timer(1.0 / 15.0, self.publish_loop)

    def publish_loop(self):
        frames = self.pipe.wait_for_frames()
        # 深度帧也需取出（管道要求），但不发布
        _ = frames.get_depth_frame()
        infra1_frame = frames.get_infrared_frame(1)
        infra2_frame = frames.get_infrared_frame(2)

        now = self.get_clock().now().to_msg()

        # 左红外
        img1 = np.asanyarray(infra1_frame.get_data())
        msg1 = self.bridge.cv2_to_imgmsg(img1, encoding="mono8")
        msg1.header.stamp = now
        msg1.header.frame_id = "camera_infra1_frame"
        self.pub_infra1.publish(msg1)

        # 右红外
        img2 = np.asanyarray(infra2_frame.get_data())
        msg2 = self.bridge.cv2_to_imgmsg(img2, encoding="mono8")
        msg2.header.stamp = now
        msg2.header.frame_id = "camera_infra2_frame"
        self.pub_infra2.publish(msg2)

    def destroy_node(self):
        self.pipe.stop()
        super().destroy_node()


def main():
    rclpy.init()
    node = DualIRPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
