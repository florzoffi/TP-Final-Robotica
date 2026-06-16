import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, ReliabilityPolicy
import csv
import math


class ScanLogger(Node):
    def __init__(self):
        super().__init__("scan_logger")

        self.output_path = "src/tpf/scans.csv"
        self.csv_file = open(self.output_path, "w", newline="")
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow([
            "time",
            "angle_min",
            "angle_increment",
            "range_min",
            "range_max",
            "ranges"
        ])

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/tb4_0/scan",
            self.scan_callback,
            qos
        )

        self.get_logger().info(f"Guardando scans en {self.output_path}")

    def scan_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        clean_ranges = []
        for r in msg.ranges:
            if math.isinf(r) or math.isnan(r):
                clean_ranges.append("")
            else:
                clean_ranges.append(f"{r:.4f}")

        ranges_str = ";".join(clean_ranges)

        self.writer.writerow([
            timestamp,
            msg.angle_min,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            ranges_str
        ])
        self.csv_file.flush()


def main(args=None):
    rclpy.init(args=args)
    node = ScanLogger()

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