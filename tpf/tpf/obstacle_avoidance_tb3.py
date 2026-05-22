import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.obstacle_detected = False
        self.rotating = False

        self.linear_speed = 0.5
        self.angular_speed = 1.0
        self.min_distance = 0.5

        self.turn_angle = math.radians(110)
        self.start_time = None
        self.rotation_time = self.turn_angle / self.angular_speed

    def scan_callback(self, msg):
        front_ranges = msg.ranges[0:20] + msg.ranges[-20:]

        valid_ranges = [
            r for r in front_ranges
            if not math.isinf(r) and not math.isnan(r)
        ]

        if len(valid_ranges) == 0:
            self.obstacle_detected = False
            return

        self.obstacle_detected = min(valid_ranges) <= self.min_distance

    def control_loop(self):
        cmd = Twist()

        if self.rotating:
            elapsed_time = self.get_clock().now().nanoseconds / 1e9 - self.start_time

            if elapsed_time < self.rotation_time:
                cmd.angular.z = self.angular_speed
            else:
                self.rotating = False
                self.start_time = None

            self.cmd_pub.publish(cmd)
            return

        if self.obstacle_detected:
            self.rotating = True
            self.start_time = self.get_clock().now().nanoseconds / 1e9
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = self.linear_speed

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()