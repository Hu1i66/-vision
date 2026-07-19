#!/usr/bin/env python3
"""Collect measured base positions from /detection_info and compute mean translation offset.

用法示例：
  source .venv/bin/activate
  source /opt/ros/humble/setup.bash
  python3 tools/collect_translation.py --ground-truth '{"x":0.2,"y":-0.2,"z":0.1}' --samples 20

输出示例（stdout）：{"dx":0.0012,"dy":-0.0234,"dz":0.0123}
该结果可直接用作 camera_to_end_effector_translation_override 的值。
"""
import argparse
import json
import sys
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except Exception:
    print('Error importing ROS2 (rclpy). Make sure you sourced ROS2 and created the virtual env correctly.', file=sys.stderr)
    raise


class Collector(Node):
    def __init__(self, ground_truth, target_samples):
        super().__init__('translation_collector')
        self.ground_truth = np.array(ground_truth, dtype=float)
        self.target = int(target_samples)
        self.samples = []
        self.sub = self.create_subscription(String, '/detection_info', self.cb, 10)
        self.get_logger().info(f'Waiting for /detection_info samples (need {self.target})')

    def cb(self, msg: String):
        try:
            data = json.loads(msg.data)
        except Exception:
            self.get_logger().warning('Invalid JSON in detection_info')
            return

        bp = data.get('base_position_m')
        if not bp:
            self.get_logger().info('detection_info has no base_position_m, skipping')
            return

        measured = np.array([bp.get('x', 0.0), bp.get('y', 0.0), bp.get('z', 0.0)], dtype=float)
        diff = self.ground_truth - measured
        self.samples.append((measured, diff))
        self.get_logger().info(f'Sample {len(self.samples)}/{self.target}  measured={measured.tolist()}  diff={diff.tolist()}')

        if len(self.samples) >= self.target:
            arr = np.array([d for (_, d) in self.samples])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            out = {'dx': float(mean[0]), 'dy': float(mean[1]), 'dz': float(mean[2])}
            self.get_logger().info(f'MEAN offset: {out}  STD: {std.tolist()}')
            print(json.dumps(out))
            rclpy.shutdown()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ground-truth', required=True, help='JSON string for ground truth base coords, e.g. "{\"x\":0.2,\"y\":-0.2,\"z\":0.1}"')
    p.add_argument('--samples', type=int, default=20, help='Number of samples to collect')
    args = p.parse_args()

    try:
        gt = json.loads(args.ground_truth)
        gt_arr = [float(gt['x']), float(gt['y']), float(gt['z'])]
    except Exception:
        print('Invalid ground-truth JSON', file=sys.stderr)
        raise

    rclpy.init()
    node = Collector(gt_arr, args.samples)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
