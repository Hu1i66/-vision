#!/usr/bin/env python3
"""
D455 同步三通道图像采集工具
=============================
同时采集 左红外 / RGB / 右红外 三张图像，确保每一组数据在同一时刻拍摄。

同步策略:
  D455 所有传感器共享同一硬件时钟，ROS2 header.stamp 即为硬件时间戳。
  对三个话题分别维护缓冲区，按时间戳匹配 (< 5ms 容差)，保证每组三帧严格同步。

用法:
  source /opt/ros/humble/setup.bash
  python3 tools/capture_synced_triple.py

  按键:
    s / Enter  — 保存当前同步组
    q / Esc    — 退出

输出目录结构:
  calib_data/
    left/    — 左红外图像  (group_000.png, group_001.png, ...)
    rgb/     — RGB 图像     (group_000.png, group_001.png, ...)
    right/   — 右红外图像  (group_000.png, group_001.png, ...)
"""

import os
import argparse
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage


# ── D455 默认话题 ──
DEFAULT_TOPICS = {
    "rgb":   "/camera/camera/color/image_raw",
    "left":  "/camera/camera/infra1/image_rect_raw",
    "right": "/camera/camera/infra2/image_rect_raw",
}

# 同步时间容差 (秒) — D455 硬件同步精度远小于此值
SYNC_TOLERANCE_S = 0.005


def rosimg_to_bgr(msg: RosImage) -> np.ndarray:
    """将 sensor_msgs/Image 转为 BGR numpy 数组 (OpenCV 格式)。"""
    h, w = msg.height, msg.width
    enc = msg.encoding.lower() if hasattr(msg, 'encoding') else ''

    # 判断通道数
    if 'mono' in enc or '8uc1' in enc:
        ch = 1
    elif '16uc1' in enc:
        ch = 1
        dtype = np.uint16
        arr = np.frombuffer(msg.data, dtype=dtype).reshape(h, w)
        # 16-bit 红外缩放到 8-bit 用于显示和保存
        arr = (arr / 256).astype(np.uint8)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    elif 'bgr' in enc:
        ch = 3
    else:
        ch = 3  # 默认 RGB

    arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, ch)

    if ch == 3:
        if 'rgb' in enc and 'bgr' not in enc:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        # bgr8 直接使用
    else:
        # 单通道灰度 → 保存原样, 显示时转 BGR
        pass

    return arr


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class SyncTripleCapture(Node):
    """订阅三个图像话题，按时间戳匹配后提供同步的图像组。"""

    def __init__(self, topics: dict, buffer_size: int = 60):
        super().__init__("sync_triple_capture")

        self.buffers = {
            "rgb":   deque(maxlen=buffer_size),
            "left":  deque(maxlen=buffer_size),
            "right": deque(maxlen=buffer_size),
        }
        self._latest_matched = None  # (ts, {rgb, left, right})

        # 订阅三个话题
        self.sub_rgb   = self.create_subscription(RosImage, topics["rgb"],   lambda m: self._cb(m, "rgb"),   10)
        self.sub_left  = self.create_subscription(RosImage, topics["left"],  lambda m: self._cb(m, "left"),  10)
        self.sub_right = self.create_subscription(RosImage, topics["right"], lambda m: self._cb(m, "right"), 10)

        self.get_logger().info(f"订阅话题:")
        self.get_logger().info(f"  RGB:   {topics['rgb']}")
        self.get_logger().info(f"  左红外: {topics['left']}")
        self.get_logger().info(f"  右红外: {topics['right']}")
        self.get_logger().info(f"同步容差: {SYNC_TOLERANCE_S*1000:.0f}ms")

    def _cb(self, msg: RosImage, key: str):
        """回调: 存入缓冲区并尝试匹配。"""
        ts = stamp_to_sec(msg.header.stamp)
        img = rosimg_to_bgr(msg)
        self.buffers[key].append((ts, img, msg))

        # 尝试匹配
        self._try_match(ts)

    def _try_match(self, trigger_ts: float):
        """在所有 buffer 中找时间戳最接近 trigger_ts 的一组帧。"""
        best = {}
        for key in ("rgb", "left", "right"):
            buf = self.buffers[key]
            if not buf:
                return
            # 找最接近 trigger_ts 的帧
            nearest = min(buf, key=lambda x: abs(x[0] - trigger_ts))
            if abs(nearest[0] - trigger_ts) > SYNC_TOLERANCE_S:
                return
            best[key] = nearest

        # 额外验证: 三帧之间两两时间差都在容差内
        timestamps = {k: v[0] for k, v in best.items()}
        pairs = [("rgb", "left"), ("rgb", "right"), ("left", "right")]
        for a, b in pairs:
            if abs(timestamps[a] - timestamps[b]) > SYNC_TOLERANCE_S:
                return

        avg_ts = sum(timestamps.values()) / 3.0
        self._latest_matched = (avg_ts, {
            "rgb":   best["rgb"][1],
            "left":  best["left"][1],
            "right": best["right"][1],
        })

    @property
    def matched_group(self):
        return self._latest_matched


def make_preview(group: dict | None, idx: int) -> np.ndarray:
    """生成预览画面: 左侧大图 RGB，右侧上下排列左右红外。"""
    if group is None:
        canvas = 255 * np.ones((540, 960, 3), dtype=np.uint8)
        cv2.putText(canvas, "Waiting for synced frames...", (30, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return canvas

    rgb = group["rgb"]
    left = group["left"]
    right = group["right"]

    # 统一显示高度
    main_h = 480
    thumb_h = main_h // 2

    # 缩放
    def resize_to_height(img, target_h):
        h, w = img.shape[:2]
        scale = target_h / h
        return cv2.resize(img, (int(w * scale), target_h))

    rgb_resized = resize_to_height(rgb, main_h)
    left_resized = resize_to_height(left, thumb_h)
    right_resized = resize_to_height(right, thumb_h)

    # 确保红外缩略图宽度一致
    thumb_w = max(left_resized.shape[1], right_resized.shape[1])
    left_resized = cv2.resize(left_resized, (thumb_w, thumb_h))
    right_resized = cv2.resize(right_resized, (thumb_w, thumb_h))

    # 画布: 左侧 rgb + 右侧两幅红外
    canvas_w = rgb_resized.shape[1] + thumb_w + 20
    canvas_h = main_h + 60  # 底部留状态栏
    canvas = 40 * np.ones((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 放置 RGB
    canvas[10:10 + main_h, 10:10 + rgb_resized.shape[1]] = rgb_resized

    # 放置左红外
    x_off = 10 + rgb_resized.shape[1] + 10
    canvas[10:10 + thumb_h, x_off:x_off + thumb_w] = left_resized
    cv2.putText(canvas, "Left IR", (x_off, 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 放置右红外
    canvas[10 + thumb_h + 6:10 + thumb_h * 2 + 6, x_off:x_off + thumb_w] = right_resized
    cv2.putText(canvas, "Right IR", (x_off, 10 + thumb_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 底部状态栏
    status_y = main_h + 35
    cv2.putText(canvas, f"Group index: {idx:04d}  |  [s/Enter] Save  [q/Esc] Quit",
                (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return canvas


def save_group(group: dict, out_dir: str, idx: int):
    """保存一组同步图像到 out_dir/{left,rgb,right}/ 下。"""
    subdirs = {
        "left":  os.path.join(out_dir, "left"),
        "rgb":   os.path.join(out_dir, "rgb"),
        "right": os.path.join(out_dir, "right"),
    }
    for sd in subdirs.values():
        os.makedirs(sd, exist_ok=True)

    fname = f"group_{idx:04d}.png"
    for key in ("left", "rgb", "right"):
        path = os.path.join(subdirs[key], fname)
        img = group[key]
        # 红外保持单通道灰度保存
        if key in ("left", "right") and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(path, img)

    print(f"Saved group_{idx:04d}: {subdirs['left']}/{fname}, {subdirs['rgb']}/{fname}, {subdirs['right']}/{fname}")


def get_next_idx(out_dir: str, prefix: str = "group") -> int:
    """自动检测已保存的最大编号，返回下一个序号。"""
    for sub in ("left", "rgb", "right"):
        d = os.path.join(out_dir, sub)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".png")]
            if files:
                max_idx = max(int(f[len(prefix)+1:len(prefix)+5]) for f in files)
                return max_idx + 1
    return 0


def parse_args():
    p = argparse.ArgumentParser(
        description="D455 同步三通道 (左红外 + RGB + 右红外) 图像采集"
    )
    p.add_argument("--out-dir", default="calib_data", help="输出根目录 (default: calib_data)")
    p.add_argument("--rgb-topic",   default=DEFAULT_TOPICS["rgb"],   help="RGB 话题")
    p.add_argument("--left-topic",  default=DEFAULT_TOPICS["left"],  help="左红外话题")
    p.add_argument("--right-topic", default=DEFAULT_TOPICS["right"], help="右红外话题")
    p.add_argument("--sync-tol-ms", type=float, default=5.0,
                   help="同步时间容差 (毫秒, default: 5.0)")
    p.add_argument("--start-index", type=int, default=None, help="起始编号 (default: 自动检测)")
    return p.parse_args()


def main():
    args = parse_args()

    global SYNC_TOLERANCE_S
    SYNC_TOLERANCE_S = args.sync_tol_ms / 1000.0

    rclpy.init()
    topics = {"rgb": args.rgb_topic, "left": args.left_topic, "right": args.right_topic}
    node = SyncTripleCapture(topics)

    idx = args.start_index if args.start_index is not None else get_next_idx(args.out_dir)
    win = "D455 Sync Capture (RGB + Stereo IR)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    print(f"\n{'='*60}")
    print(f"  D455 同步三通道图像采集")
    print(f"  输出目录: {os.path.abspath(args.out_dir)}")
    print(f"  起始编号: {idx:04d}")
    print(f"  按键: [s/Enter] 保存当前组  [q/Esc] 退出")
    print(f"{'='*60}\n")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            ts, group = node.matched_group if node.matched_group else (None, None)
            preview = make_preview(group, idx)

            # 显示时间戳差值
            if group is not None:
                c_h = preview.shape[0]
                cv2.putText(preview, f"Synced | ts={ts:.3f}",
                            (preview.shape[1] - 300, c_h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

            cv2.imshow(win, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (13, 10, ord('s')):  # Enter or 's'
                if group is not None:
                    save_group(group, args.out_dir, idx)
                    idx += 1
                else:
                    print("[WARN] 尚未收到同步图像组，无法保存")

            elif key == ord('q') or key == 27:  # q or Esc
                print("用户退出")
                break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(f"\n共采集 {idx - (args.start_index if args.start_index is not None else 0)} 组图像")
    print(f"数据保存在: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
