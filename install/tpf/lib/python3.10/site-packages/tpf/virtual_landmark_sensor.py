import csv
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class VirtualLandmarkSensor(Node):

    def __init__(self):
        super().__init__("virtual_landmark_sensor")

        self.declare_parameter("landmark_csv", "")
        self.declare_parameter("max_range", 2.5)
        self.declare_parameter("fov", 2.1)  # aprox 120 grados
        self.declare_parameter("publish_rate", 5.0)

        self.landmark_csv = self.get_parameter("landmark_csv").value
        self.max_range = float(self.get_parameter("max_range").value)
        self.fov = float(self.get_parameter("fov").value)

        self.robot_pose = None

        self.map_received = False
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.occupancy_data = None

        self.landmarks = self.load_landmarks(self.landmark_csv)

        self.pose_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos,
        )

        self.obs_pub = self.create_publisher(
            PoseArray,
            "/aruco_observations",
            10,
        )

        self.marker_pub = self.create_publisher(
            PoseArray,
            "/virtual_landmarks",
            10,
        )

        period = 1.0 / float(self.get_parameter("publish_rate").value)
        self.timer = self.create_timer(period, self.publish_observations)

        self.get_logger().info(
            f"Virtual landmark sensor iniciado con {len(self.landmarks)} landmarks."
        )

    def load_landmarks(self, path):
        landmarks = []

        if not path:
            self.get_logger().warn("No se paso landmark_csv.")
            return landmarks

        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    tag_id = int(row["id"])
                    x = float(row["x"])
                    y = float(row["y"])
                    landmarks.append((tag_id, x, y))
        except Exception as e:
            self.get_logger().error(f"No pude cargar landmarks desde {path}: {e}")

        return landmarks

    def odom_callback(self, msg):

        self.robot_pose = PoseStamped()
        self.robot_pose.header = msg.header
        self.robot_pose.pose = msg.pose.pose

        
    def map_callback(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.occupancy_data = list(msg.data)
        self.map_received = True

    def world_to_map(self, x, y):
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None

        return row, col

    def is_occupied_world(self, x, y):
        cell = self.world_to_map(x, y)
        if cell is None:
            return True

        row, col = cell
        idx = row * self.map_width + col
        occ = self.occupancy_data[idx]

        return occ > 50 or occ == -1

    def has_line_of_sight(self, x0, y0, x1, y1):
        """
        Devuelve False si hay pared entre robot y landmark.
        """
        dist = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

        if dist < 1e-6:
            return True

        step = self.map_resolution * 0.5
        n = max(1, int(dist / step))

        for i in range(1, n):
            t = i / n
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)

            if self.is_occupied_world(x, y):
                return False

        return True

    def publish_landmark_markers(self, stamp):
        msg = PoseArray()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"

        for tag_id, x, y in self.landmarks:
            p = Pose()
            p.position.x = x
            p.position.y = y
            p.position.z = 0.0
            p.orientation.w = 1.0
            msg.poses.append(p)

        self.marker_pub.publish(msg)

    def publish_observations(self):
        if self.robot_pose is None or not self.map_received:
            return

        
        rx = self.robot_pose.pose.position.x
        ry = self.robot_pose.pose.position.y
        ryaw = yaw_from_quaternion(self.robot_pose.pose.orientation)

        obs = PoseArray()
        obs.header.stamp = self.get_clock().now().to_msg()
        obs.header.frame_id = "base_link"

        for tag_id, lx, ly in self.landmarks:
            dx = lx - rx
            dy = ly - ry

            distance = math.sqrt(dx * dx + dy * dy)
            if distance > self.max_range:
                continue

            bearing = normalize_angle(math.atan2(dy, dx) - ryaw)
            if abs(bearing) > self.fov / 2.0:
                continue

            if not self.has_line_of_sight(rx, ry, lx, ly):
                continue

            p = Pose()
            p.position.x = float(tag_id)
            p.position.y = float(distance)
            p.position.z = float(bearing)
            obs.poses.append(p)
            
        self.obs_pub.publish(obs)
        self.publish_landmark_markers(obs.header.stamp)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualLandmarkSensor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()