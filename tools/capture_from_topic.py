#!/usr/bin/env python3
"""从 ROS2 图像话题或本地摄像头显示画面，按 Enter 保存图片（并生成空的 YOLO 标签文件）。

用法示例：
  # ROS 模式（默认）
  python3 tools/capture_from_topic.py --out-dir images/train --labels-dir labels/train

  # 本地摄像头模式（调试用）
  python3 tools/capture_from_topic.py --mode cam --cam-index 0
"""
import os
import argparse
# sys 可选保留以便未来扩展
import cv2
import numpy as np


def next_index_for_prefix(out_dir, prefix, pad=3):
    if not os.path.exists(out_dir):
        return 0
    files = [f for f in os.listdir(out_dir) if f.startswith(prefix) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    max_idx = -1
    for f in files:
        name = os.path.splitext(f)[0]
        parts = name.rsplit('_', 1)
        if len(parts) == 2 and parts[0] == prefix:
            try:
                idx = int(parts[1])
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                continue
    return max_idx + 1


def save_image_and_label(img, out_dir, labels_dir, prefix, idx, img_fmt='jpg'):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    fname = f"{prefix}_{idx:03d}.{img_fmt}"
    fpath = os.path.join(out_dir, fname)
    cv2.imwrite(fpath, img)
    # 创建空的 YOLO 标签文件（等待人工标注）
    label_name = os.path.splitext(fname)[0] + '.txt'
    label_path = os.path.join(labels_dir, label_name)
    open(label_path, 'w').close()
    print(f"Saved: {fpath}  label: {label_path}")
    return fpath, label_path


def run_cam_mode(args):
    cap = cv2.VideoCapture(args.cam_index)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.cam_index}")
        return

    idx = args.start_index if args.start_index is not None else next_index_for_prefix(args.out_dir, args.prefix)
    win = 'Capture'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to read frame from camera')
                break

            disp = frame.copy()
            cv2.putText(disp, f'Idx={idx:03d}  Enter=save  Q=quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):
                save_image_and_label(frame, args.out_dir, args.labels_dir, args.prefix, idx)
                idx += 1
            elif key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run_ros_mode(args):
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image as RosImage
    except Exception as e:
        print('ROS2 Python package (rclpy) not available or not sourced.\n', e)
        print('如果想直接用本地摄像头，请使用 --mode cam')
        return

    class ImgSaverNode(Node):
        def __init__(self, topic):
            super().__init__('image_capture_node')
            self.latest = None
            self.create_subscription(RosImage, topic, self.cb, 10)
            self.get_logger().info(f'Subscribed to {topic}')

        def cb(self, msg: RosImage):
            try:
                h = msg.height
                w = msg.width
                step = getattr(msg, 'step', None)
                arr = np.frombuffer(msg.data, dtype=np.uint8)
                # 计算通道数
                if step and w:
                    ch = int(step // w)
                else:
                    # 兜底为3
                    ch = 3
                try:
                    img = arr.reshape(h, w, ch)
                except Exception:
                    # 尝试按3通道reshape
                    img = arr.reshape(h, w, 3)

                enc = getattr(msg, 'encoding', '').lower() if hasattr(msg, 'encoding') else ''
                if ch == 3:
                    if 'rgb' in enc and 'bgr' not in enc:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif 'bgr' in enc:
                        pass
                    else:
                        # 默认按 RGB->BGR
                        try:
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        except Exception:
                            pass
                elif ch == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                self.latest = img
            except Exception as e:
                self.get_logger().warning(f'convert image fail: {e}')

    rclpy.init()
    node = ImgSaverNode(args.topic)
    idx = args.start_index if args.start_index is not None else next_index_for_prefix(args.out_dir, args.prefix)
    win = 'Capture'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest is None:
                disp = 255 * np.ones((360, 640, 3), dtype=np.uint8)
                cv2.putText(disp, 'Waiting for frames...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                img = disp
            else:
                img = node.latest.copy()

            cv2.putText(img, f'Idx={idx:03d}  Enter=save  Q=quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):
                if node.latest is not None:
                    save_image_and_label(node.latest, args.out_dir, args.labels_dir, args.prefix, idx)
                    idx += 1
                else:
                    print('No frame to save yet')
            elif key == ord('q') or key == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description='Capture images from ROS2 topic or local camera (Enter to save)')
    p.add_argument('--mode', choices=('ros', 'cam'), default='ros', help='运行模式：ros（订阅话题） 或 cam（本地摄像头）')
    p.add_argument('--topic', default='/camera/camera/color/image_raw', help='ROS 图像话题')
    p.add_argument('--out-dir', default='images/train', help='保存图片目录')
    p.add_argument('--labels-dir', default='labels/train', help='保存标签目录')
    p.add_argument('--prefix', default='orange', help='文件名前缀，例如 orange')
    p.add_argument('--start-index', type=int, default=None, help='起始编号（覆盖自动检测）')
    p.add_argument('--cam-index', type=int, default=0, help='本地摄像头设备索引（mode=cam 时生效）')
    return p.parse_args()


def main():
    args = parse_args()
    # 预创建目录
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.labels_dir, exist_ok=True)

    if args.mode == 'cam':
        run_cam_mode(args)
    else:
        run_ros_mode(args)


if __name__ == '__main__':
    main()
