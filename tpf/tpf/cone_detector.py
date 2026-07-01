import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rcl_interfaces.msg import SetParametersResult

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class ConeDetector(Node):
    """
    Detecta conos rojos en la imagen de la camara (segmentacion HSV),
    estima su distancia (por tamano aparente) y bearing, y publica la
    coordenada estimada en frame "map" sobre /cone_detection una vez que
    la deteccion esta confirmada (varios frames consecutivos consistentes).

    Nunca mueve el robot ni publica /goal_pose directamente — eso es
    responsabilidad de cone_mission_manager, que ademas valida la
    coordenada contra el mapa de ocupacion antes de navegar hacia ella.
    """

    def __init__(self):
        super().__init__("cone_detector")

        self.bridge = CvBridge()

        # ---------------- parametros (tuneables sin recompilar) ----------------
        self.declare_parameter("hue_low_max", 10)
        self.declare_parameter("hue_high_min", 170)
        self.declare_parameter("sat_min", 120)
        self.declare_parameter("val_min", 80)
        self.declare_parameter("min_contour_area_px", 80.0)
        self.declare_parameter("min_aspect_ratio", 1.2)
        self.declare_parameter("max_aspect_ratio", 10.0)
        self.declare_parameter("morph_kernel_size", 3)
        self.declare_parameter("apply_morph_open", False)
        # Factor de escala para la estimacion de distancia por tamaño aparente.
        # No es necesariamente la altura fisica del cono — en la practica
        # representa la altura efectiva de la region segmentada (mascara HSV)
        # que la camara ve, que depende del angulo de vision, la iluminacion
        # y los umbrales de color. Se calibra empiricamente: se ajusta hasta
        # que la coordenada publicada en /cone_detection coincide con la
        # posicion real del cono en el mapa.
        self.declare_parameter("cone_distance_scale_m", 0.50)
        self.declare_parameter("min_consistent_detections", 3)
        self.declare_parameter("max_spread_m", 0.30)
        self.declare_parameter("max_age_sec", 2.0)
        self.declare_parameter("recompute_distance_m", 0.5)
        self.declare_parameter("debug_publish", True)
        self.declare_parameter("debug_cv_window", False)

        self.hue_low_max = self.get_parameter("hue_low_max").value
        self.hue_high_min = self.get_parameter("hue_high_min").value
        self.sat_min = self.get_parameter("sat_min").value
        self.val_min = self.get_parameter("val_min").value
        self.min_contour_area_px = self.get_parameter("min_contour_area_px").value
        self.min_aspect_ratio = self.get_parameter("min_aspect_ratio").value
        self.max_aspect_ratio = self.get_parameter("max_aspect_ratio").value
        self.morph_kernel_size = self.get_parameter("morph_kernel_size").value
        self.apply_morph_open = self.get_parameter("apply_morph_open").value
        self.cone_distance_scale_m = self.get_parameter("cone_distance_scale_m").value
        self.min_consistent_detections = self.get_parameter("min_consistent_detections").value
        self.max_spread_m = self.get_parameter("max_spread_m").value
        self.max_age_sec = self.get_parameter("max_age_sec").value
        self.recompute_distance_m = self.get_parameter("recompute_distance_m").value
        self.debug_publish = self.get_parameter("debug_publish").value
        self.debug_cv_window = self.get_parameter("debug_cv_window").value

        # "ros2 param set" solo actualiza el valor en el servidor de
        # parametros — sin este callback, los atributos cacheados arriba
        # (los que realmente usa segment_red/find_best_cone_contour) nunca
        # se enterarian del cambio y el tuneo en vivo no tendria efecto.
        self.add_on_set_parameters_callback(self.on_parameters_set)

        # ---------------- estado de calibracion / pose ----------------
        self.camera_matrix = None  # se completa con el primer /camera_info
        self.current_pose = None

        # ---------------- estado de debounce ----------------
        # cada entrada: (x, y, stamp_sec)
        self.recent_detections = deque(maxlen=20)
        self.last_confirmed_xy = None

        # ---------------- subscripciones ----------------
        qos_camera = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

        self.image_sub = self.create_subscription(
            Image,
            "/tb4_0/oakd/rgb/preview/image_raw",
            self.image_callback,
            qos_camera,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/tb4_0/oakd/rgb/preview/camera_info",
            self.camera_info_callback,
            qos_camera,
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/estimated_pose",
            self.pose_callback,
            10,
        )

        # ---------------- publicaciones ----------------
        self.cone_pub = self.create_publisher(PointStamped, "/cone_detection", 10)

        if self.debug_publish:
            self.debug_image_pub = self.create_publisher(Image, "/cone_detector/debug_image", 10)
            self.debug_mask_pub = self.create_publisher(Image, "/cone_detector/debug_mask", 10)

        self.get_logger().info(
            "Cone detector iniciado. Esperando /camera_info para calibrar..."
        )

    # ------------------------------------------------------------------
    def on_parameters_set(self, params):
        """
        Aplica en caliente los cambios hechos con "ros2 param set", para
        poder tunear HSV/area/aspect-ratio mientras se mira el debug_mask
        en rviz, sin tener que relanzar el nodo.
        """
        for param in params:
            if hasattr(self, param.name):
                setattr(self, param.name, param.value)
        return SetParametersResult(successful=True)

    # ------------------------------------------------------------------
    def camera_info_callback(self, msg):
        if self.camera_matrix is not None:
            return
        k = msg.k
        self.camera_matrix = np.array(
            [[k[0], k[1], k[2]],
             [k[3], k[4], k[5]],
             [k[6], k[7], k[8]]],
            dtype=np.float64,
        )
        self.get_logger().info(
            f"Camera info recibida — fx={k[0]:.2f} fy={k[4]:.2f} cx={k[2]:.2f} cy={k[5]:.2f}"
        )

    def pose_callback(self, msg):
        self.current_pose = msg

    # ------------------------------------------------------------------
    def image_callback(self, msg):
        # La segmentacion y las imagenes de debug no dependen de calibracion
        # ni de pose — se generan siempre, para poder tunear HSV/area/aspect
        # ratio mirando /cone_detector/debug_image y /cone_detector/debug_mask
        # en rviz aunque todavia no haya /initialpose ni /camera_info.
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        mask = self.segment_red(frame)

        candidate = self.find_best_cone_contour(mask)

        debug_frame = frame
        if candidate is not None:
            x, y, w, h = candidate
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_frame, (x + w // 2, y + h // 2), 4, (0, 255, 0), -1)

            # La estimacion de distancia/bearing y la proyeccion al frame
            # "map" si necesitan calibracion (camera_info) y pose
            # (estimated_pose) — recien disponibles una vez que el FSM de
            # Parte B esta localizado.
            if self.camera_matrix is None:
                self.get_logger().warn(
                    "Cono visto pero sin /camera_info todavia — no se puede estimar distancia.",
                    throttle_duration_sec=5.0,
                )
            elif self.current_pose is None:
                self.get_logger().warn(
                    "Cono visto pero sin /estimated_pose todavia — no se puede ubicar en el mapa.",
                    throttle_duration_sec=5.0,
                )
            else:
                distance, bearing = self.estimate_distance_bearing(x, w, h)
                cone_x, cone_y = self.camera_to_map(distance, bearing)

                timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.register_candidate(cone_x, cone_y, timestamp)

                self.get_logger().info(
                    f"Candidato cono: dist={distance:.2f}m bearing={math.degrees(bearing):.1f}deg "
                    f"map=({cone_x:.2f},{cone_y:.2f})",
                    throttle_duration_sec=1.0,
                )

        if self.debug_publish:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            mask_msg.header = msg.header
            self.debug_mask_pub.publish(mask_msg)

        if self.debug_cv_window:
            cv2.imshow("Cone detector", debug_frame)
            cv2.waitKey(1)

    # ------------------------------------------------------------------
    def segment_red(self, frame):
        """
        Segmentacion HSV de rojo. El rojo cruza el limite 0/180 del canal H
        de OpenCV, asi que usamos dos rangos y los combinamos con OR.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_a = np.array([0, self.sat_min, self.val_min], dtype=np.uint8)
        upper_a = np.array([self.hue_low_max, 255, 255], dtype=np.uint8)

        lower_b = np.array([self.hue_high_min, self.sat_min, self.val_min], dtype=np.uint8)
        upper_b = np.array([179, 255, 255], dtype=np.uint8)

        mask_a = cv2.inRange(hsv, lower_a, upper_a)
        mask_b = cv2.inRange(hsv, lower_b, upper_b)
        mask = cv2.bitwise_or(mask_a, mask_b)

        # OPEN (erode+dilate) saca ruido chico, pero tambien erosiona el
        # cuerpo angosto/en punta de un cono lejano — con un kernel grande
        # puede partirlo en fragmentos que individualmente no pasan los
        # filtros de area/aspect-ratio (intermitencia que mejora con la
        # distancia, justamente porque el cono ocupa mas pixeles). Por eso
        # esta deshabilitado por default; el filtro de area ya cubre la
        # funcion de "sacar ruido chico" sin destruir formas finas.
        kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
        if self.apply_morph_open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def find_best_cone_contour(self, mask):
        """
        Entre los contornos que pasan los filtros de area y aspect ratio,
        devuelve el bounding box (x, y, w, h) del de mayor area (el
        candidato mas confiable / probablemente mas cercano).
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.get_logger().info(
                "DIAG: ningún contorno en la mascara (mask completamente negra o filtro HSV demasiado estricto).",
                throttle_duration_sec=2.0,
            )
            return None

        # Log del contorno mas grande (independientemente de si pasa los filtros)
        # para entender qué hay en la mascara antes de aplicar cualquier criterio.
        areas = [cv2.contourArea(c) for c in contours]
        biggest_idx = int(max(range(len(areas)), key=lambda i: areas[i]))
        bx, by, bw, bh = cv2.boundingRect(contours[biggest_idx])
        biggest_ar = (bh / bw) if bw > 0 else 0.0
        self.get_logger().info(
            f"DIAG: {len(contours)} contorno(s) | mayor: area={areas[biggest_idx]:.0f}px "
            f"w={bw} h={bh} aspect={biggest_ar:.2f} "
            f"(umbral area>={self.min_contour_area_px:.0f}, "
            f"aspect [{self.min_aspect_ratio:.1f},{self.max_aspect_ratio:.1f}])",
            throttle_duration_sec=1.0,
        )

        best = None
        best_area = 0.0

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_contour_area_px:
                continue

            x, y, w, h = cv2.boundingRect(c)
            if w == 0:
                continue

            aspect_ratio = h / w
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                self.get_logger().info(
                    f"DIAG: contorno area={area:.0f} RECHAZADO por aspect_ratio={aspect_ratio:.2f} "
                    f"(fuera de [{self.min_aspect_ratio:.1f},{self.max_aspect_ratio:.1f}])",
                    throttle_duration_sec=1.0,
                )
                continue

            if area > best_area:
                best_area = area
                best = (x, y, w, h)

        if best is None:
            self.get_logger().info(
                "DIAG: ningun contorno paso todos los filtros.",
                throttle_duration_sec=1.0,
            )

        return best

    # ------------------------------------------------------------------
    def estimate_distance_bearing(self, x, w, h):
        """
        Distancia por tamano aparente (no hay marcador ArUco en el cono):
        distance = altura_real_m * fy / altura_px

        Bearing por offset del centro del bounding box respecto al centro
        optico de la imagen, mismo signo que aruco_detector.py.
        """
        fx = self.camera_matrix[0][0]
        fy = self.camera_matrix[1][1]
        cx_principal = self.camera_matrix[0][2]

        distance = (self.cone_distance_scale_m * fy) / float(h)

        cx_box = x + w / 2.0
        pixel_offset = cx_box - cx_principal
        bearing = -math.atan2(pixel_offset, fx)

        return distance, bearing

    def camera_to_map(self, distance, bearing):
        """
        Simplificacion documentada: se asume que la camara esta montada
        mirando hacia adelante, sin offset respecto al origen de base_link.
        El error introducido (unos pocos cm) es menor que el propio error
        del modelo de distancia por tamano aparente.
        """
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        robot_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)

        world_angle = robot_yaw + bearing
        cone_x = robot_x + distance * math.cos(world_angle)
        cone_y = robot_y + distance * math.sin(world_angle)

        return cone_x, cone_y

    # ------------------------------------------------------------------
    def register_candidate(self, x, y, timestamp):
        """
        Mantiene un buffer de las ultimas detecciones (con timestamp) y
        confirma (publica) una deteccion solo cuando hay suficientes
        candidatos recientes y mutuamente consistentes — evita publicar
        ruido frame a frame.
        """
        self.recent_detections.append((x, y, timestamp))

        cutoff = timestamp - self.max_age_sec
        while self.recent_detections and self.recent_detections[0][2] < cutoff:
            self.recent_detections.popleft()

        if len(self.recent_detections) < self.min_consistent_detections:
            return

        xs = [d[0] for d in self.recent_detections]
        ys = [d[1] for d in self.recent_detections]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)

        spread = max(
            math.hypot(px - mean_x, py - mean_y) for px, py in zip(xs, ys)
        )

        if spread > self.max_spread_m:
            return

        # Ya confirmamos este mismo cluster antes — no repetir publicacion
        # salvo que el cluster se haya movido (otro cono, o correccion).
        if self.last_confirmed_xy is not None:
            dx = mean_x - self.last_confirmed_xy[0]
            dy = mean_y - self.last_confirmed_xy[1]
            if math.hypot(dx, dy) < self.recompute_distance_m:
                return

        self.last_confirmed_xy = (mean_x, mean_y)
        self.publish_confirmed(mean_x, mean_y)

    def publish_confirmed(self, x, y):
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.point.x = x
        msg.point.y = y
        msg.point.z = 0.0
        self.cone_pub.publish(msg)

        self.get_logger().info(f"Cono confirmado en mapa: ({x:.2f}, {y:.2f})")


def main(args=None):
    rclpy.init(args=args)

    node = ConeDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
