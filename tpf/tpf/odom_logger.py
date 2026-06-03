import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class OdomLogger(Node):
    def __init__(self):
        super().__init__("odom_logger")

        self.output_path = "src/tpf/odom.csv"
        self.csv_file = open(self.output_path, "w")
        self.csv_file.write("time,x,y,theta\n")
        self.csv_file.flush()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/tb4_0/odom",
            self.odom_callback,
            qos
        )

        self.get_logger().info(f"Guardando odometria en {self.output_path}")

    def odom_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = quaternion_to_yaw(msg.pose.pose.orientation)

        self.csv_file.write(f"{timestamp},{x:.6f},{y:.6f},{theta:.6f}\n")
        self.csv_file.flush()


def main(args=None):
    rclpy.init(args=args)

    node = OdomLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()