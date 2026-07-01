import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
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

        self.image_pub = self.create_publisher(
            Image,
            "/aruco_image",
            10
        )

        # Publica observaciones en tiempo real para el filtro de particulas.
        # Cada Pose codifica: position.x=tag_id, position.y=distance, position.z=bearing.
        self.obs_pub = self.create_publisher(PoseArray, "/aruco_observations", 10)
        self.declare_parameter("save_csv", True)
        self.save_csv = self.get_parameter("save_csv").value

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_100
        )
        self.aruco_params = cv2.aruco.DetectorParameters()

        #self.aruco_params.adaptiveThreshWinSizeMin = 3
        #self.aruco_params.adaptiveThreshWinSizeMax = 45
        #self.aruco_params.adaptiveThreshWinSizeStep = 4
#
        #self.aruco_params.minMarkerPerimeterRate = 0.03
        #self.aruco_params.maxMarkerPerimeterRate = 4.0
#
        #self.aruco_params.polygonalApproxAccuracyRate = 0.03
        #self.aruco_params.minCornerDistanceRate = 0.03
        #self.aruco_params.minDistanceToBorder = 3
#
        #self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        #self.aruco_params.cornerRefinementWinSize = 5
        #self.aruco_params.cornerRefinementMaxIterations = 30
        #self.aruco_params.cornerRefinementMinAccuracy = 0.01

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

        self.output_path = "src/TP-Final-Robotica/tpf/aruco_observations.csv"

        if self.save_csv:
            self.csv_file = open(self.output_path, "w")
            self.csv_file.write("time,tag_id,distance,bearing\n")
            self.csv_file.flush()
            self.get_logger().info(f"Guardando observaciones en {self.output_path}")
        else:
            self.csv_file = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #clahe = cv2.createCLAHE(
        #    clipLimit=2.0,
        #    tileGridSize=(8, 8)
        #)
        #gray = cv2.GaussianBlur(gray, (3, 3), 0)

        corners, ids, rejected = self.detector.detectMarkers(gray)

        obs_array = PoseArray()
        obs_array.header = msg.header
        obs_array.header.frame_id = "camera"

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.camera_matrix,
                self.dist_coeffs
            )
            ids_flat = ids.flatten()
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(0, 0, 255))

            for i, tag_id in enumerate(ids_flat):
                tvec = tvecs[i][0]

                x = tvec[0]
                y = tvec[1]
                z = tvec[2]

                distance = np.sqrt(x**2 + z**2)
                bearing = -np.arctan2(x, z)

                if distance < 0.2 or distance > 1.5:
                    continue

                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                if self.save_csv:
                    self.csv_file.write(
                        f"{timestamp},{int(tag_id)},{distance:.6f},{bearing:.6f}\n"
                    )
                    self.csv_file.flush()

                self.get_logger().info(
                    f"Tag {tag_id} | dist={distance:.2f}m | bearing={bearing:.2f}rad"
                )

                # Codificamos la observacion en un Pose estandar:
                # position.x = tag_id, position.y = distance, position.z = bearing
                p = Pose()
                p.position.x = float(tag_id)
                p.position.y = float(distance)
                p.position.z = float(bearing)
                obs_array.poses.append(p)

        self.obs_pub.publish(obs_array)

        annotated_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        annotated_msg.header = msg.header
        self.image_pub.publish(annotated_msg)


def main(args=None):
    rclpy.init(args=args)

    node = ArucoDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.csv_file is not None:
            node.csv_file.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()