import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import os

class ArucoDetector(Node):
    def __init__(self):
        super().__init__("aruco_detector")

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/tb4_0/oakd/rgb/preview/image_raw",
            self.image_callback,
            10
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dict,
            self.aruco_params
        )

        self.get_logger().info("Aruco detector node started")
        self.marker_size = 0.0889

        self.camera_matrix = np.array([
            [203.14, 0.0, 122.57],
            [0.0, 361.13, 123.33],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        self.dist_coeffs = np.array([
            -0.9904393553733826,
            -47.16939926147461,
            -0.0007601691759191453,
            -0.00031758102704770863,
            306.0343933105469
        ], dtype=np.float64)

        self.output_path = "src/tpf/aruco_observations.csv"
        self.csv_file = open(self.output_path, "w")
        self.csv_file.write("time,tag_id,distance,bearing\n")
        self.csv_file.flush()
        self.get_logger().info(f"Guardando observaciones en {self.output_path}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.camera_matrix,
                self.dist_coeffs
            )
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for i, tag_id in enumerate( ids_flat ):
                tvec = tvecs[i][0]

                x = tvec[0]
                y = tvec[1]
                z = tvec[2]

                distance = np.sqrt(x**2 + z**2)
                bearing = np.arctan2(x, z)

                if distance < 0.2 or distance > 2.0:
                    continue
                
                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                self.csv_file.write(
                    f"{timestamp},{int(tag_id)},{distance:.6f},{bearing:.6f}\n"
                )
                self.csv_file.flush()

                self.get_logger().info(
                    f"Tag {tag_id} | dist={distance:.2f}m | bearing={bearing:.2f}rad"
                )

        cv2.imshow("Aruco detections", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)

    node = ArucoDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()