import numpy as np
import pandas as pd

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import matplotlib.pyplot as plt

RESOLUTION = 0.05
MAP_SIZE = 30.0
GRID_N = int(MAP_SIZE / RESOLUTION)

ORIGIN_X = MAP_SIZE / 2.0
ORIGIN_Y = MAP_SIZE / 2.0

BEAM_STEP = 20
MAX_RANGE = 2.0

MIN_OCC_HITS = 4
MIN_FREE_HITS = 2
LIDAR_ANGLE_OFFSET = np.pi / 2   # probá: 0, np.pi/2, -np.pi/2, np.pi

POSES_CSV = "src/tpf/poses_optimized_keyframes.csv"


def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_to_quat(yaw):
    qz = np.sin(yaw / 2.0)
    qw = np.cos(yaw / 2.0)
    return 0.0, 0.0, qz, qw


def world_to_grid(x, y):
    gx = int((x + ORIGIN_X) / RESOLUTION)
    gy = int((y + ORIGIN_Y) / RESOLUTION)
    return gx, gy


def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0

    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return points


class CorrectedMapNode(Node):
    def __init__(self):
        super().__init__("corrected_map_node")

        poses = pd.read_csv(POSES_CSV)
        self.times = poses["time"].to_numpy()
        self.xs = poses["x"].to_numpy()
        self.ys = poses["y"].to_numpy()
        self.thetas = np.unwrap(poses["theta"].to_numpy())

        self.occ_count = np.zeros((GRID_N, GRID_N), dtype=np.uint16)
        self.free_count = np.zeros((GRID_N, GRID_N), dtype=np.uint16)

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", 10)
        self.belief_pub = self.create_publisher(PoseStamped, "/belief", 10)
        self.path_pub = self.create_publisher(Path, "/poses_guardadas", 10)

        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/tb4_0/scan",
            self.scan_callback,
            qos
        )

        self.get_logger().info("Corrected map node started")
        self.tf_broadcaster = TransformBroadcaster(self)

    def interpolate_pose(self, t):
        if t < self.times[0] or t > self.times[-1]:
            return None

        idx = np.searchsorted(self.times, t)

        if idx == 0:
            return self.xs[0], self.ys[0], normalize_angle(self.thetas[0])

        if idx >= len(self.times):
            return self.xs[-1], self.ys[-1], normalize_angle(self.thetas[-1])

        t0 = self.times[idx - 1]
        t1 = self.times[idx]

        alpha = (t - t0) / (t1 - t0)

        x = (1 - alpha) * self.xs[idx - 1] + alpha * self.xs[idx]
        y = (1 - alpha) * self.ys[idx - 1] + alpha * self.ys[idx]
        theta = (1 - alpha) * self.thetas[idx - 1] + alpha * self.thetas[idx]

        return x, y, normalize_angle(theta)
    
    def scan_callback(self, scan):
        t = scan.header.stamp.sec + scan.header.stamp.nanosec * 1e-9
        nearest_error = np.min(np.abs(self.times - t))
        if nearest_error > 0.25:
            return

        pose = self.interpolate_pose(t)
        if pose is None:
            return

        x, y, theta = pose

        self.publish_belief(scan.header, x, y, theta)

        robot_gx, robot_gy = world_to_grid(x, y)
        if not (0 <= robot_gx < GRID_N and 0 <= robot_gy < GRID_N):
            return

        range_max = min(scan.range_max, MAX_RANGE)

        for k in range(0, len(scan.ranges), BEAM_STEP):
            r = scan.ranges[k]

            if np.isnan(r) or np.isinf(r):
                continue
            if r < scan.range_min or r > range_max:
                continue

            angle = theta + LIDAR_ANGLE_OFFSET + scan.angle_min + k * scan.angle_increment

            wx = x + r * np.cos(angle)
            wy = y + r * np.sin(angle)

            end_gx, end_gy = world_to_grid(wx, wy)

            if not (0 <= end_gx < GRID_N and 0 <= end_gy < GRID_N):
                continue

            ray = bresenham(robot_gx, robot_gy, end_gx, end_gy)

            for gx, gy in ray[:-1]:
                row = gy
                col = gx
                if 0 <= row < GRID_N and 0 <= col < GRID_N:
                    self.free_count[row, col] += 1

            gx, gy = ray[-1]
            row = gy
            col = gx
            if 0 <= row < GRID_N and 0 <= col < GRID_N:
                self.occ_count[row, col] += 1

        self.publish_map(scan.header)

    def publish_belief(self, header, x, y, theta):
        msg = PoseStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "map"

        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = 0.0

        qx, qy, qz, qw = yaw_to_quat(theta)
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw

        self.belief_pub.publish(msg)

        self.path_msg.header.stamp = header.stamp
        self.path_msg.poses.append(msg)
        self.path_pub.publish(self.path_msg)
        tf = TransformStamped()
        tf.header.stamp = header.stamp
        tf.header.frame_id = "map"
        tf.child_frame_id = "rplidar_link"
        
        tf.transform.translation.x = float(x)
        tf.transform.translation.y = float(y)
        tf.transform.translation.z = 0.0
        
        qx_l, qy_l, qz_l, qw_l = yaw_to_quat(theta + LIDAR_ANGLE_OFFSET)

        tf.transform.rotation.x = qx_l
        tf.transform.rotation.y = qy_l
        tf.transform.rotation.z = qz_l
        tf.transform.rotation.w = qw_l
        
        self.tf_broadcaster.sendTransform(tf)

    def publish_map(self, header):
        free_mask = self.free_count >= MIN_FREE_HITS
        occ_mask = (self.occ_count >= MIN_OCC_HITS) & (self.occ_count > self.free_count * 0.25)

        grid = np.full((GRID_N, GRID_N), -1, dtype=np.int8)
        grid[free_mask] = 0
        grid[occ_mask] = 100

        msg = OccupancyGrid()
        msg.header.stamp = header.stamp
        msg.header.frame_id = "map"

        msg.info.resolution = RESOLUTION
        msg.info.width = GRID_N
        msg.info.height = GRID_N

        msg.info.origin.position.x = -ORIGIN_X
        msg.info.origin.position.y = -ORIGIN_Y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0

        msg.data = grid.flatten().tolist()

        self.map_pub.publish(msg)
        map_img = np.full((GRID_N, GRID_N), 127, dtype=np.uint8)
        map_img[free_mask] = 255
        map_img[occ_mask] = 0
        
        plt.imsave("src/tpf/final_map.png", map_img, cmap="gray")


def main(args=None):
    rclpy.init(args=args)
    node = CorrectedMapNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()