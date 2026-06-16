import pandas as pd
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


LANDMARKS_CSV = "src/tpf/landmarks_optimized_keyframes.csv"


class LandmarksPublisher(Node):
    def __init__(self):
        super().__init__("landmarks_publisher")

        self.landmarks = pd.read_csv(LANDMARKS_CSV)
        self.pub = self.create_publisher(MarkerArray, "/landmarks", 10)
        self.timer = self.create_timer(1.0, self.publish_landmarks)

        self.get_logger().info("Landmarks publisher started")

    def publish_landmarks(self):
        marker_array = MarkerArray()

        for i, row in self.landmarks.iterrows():
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.ns = "aruco_landmarks"
            marker.id = int(row["tag_id"])
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(row["x"])
            marker.pose.position.y = float(row["y"])
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.15
            marker.scale.y = 0.15
            marker.scale.z = 0.15

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()

            text.ns = "aruco_labels"
            text.id = int(row["tag_id"]) + 1000
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD

            text.pose.position.x = float(row["x"])
            text.pose.position.y = float(row["y"])
            text.pose.position.z = 0.25
            text.pose.orientation.w = 1.0

            text.scale.z = 0.2

            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0

            text.text = str(int(row["tag_id"]))

            marker_array.markers.append(text)

        self.pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LandmarksPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()