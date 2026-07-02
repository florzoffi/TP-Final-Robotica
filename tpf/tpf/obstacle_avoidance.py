import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math


class ObstacleAvoidance(Node):
    """
    Detecta obstáculos en el cono frontal y señaliza a la FSM.
    Estrategia: frenar + esperar a que el planner (con datos de LIDAR) genere
    un camino alternativo. No gira ni avanza por su cuenta.
    """

    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.declare_parameter('robot_type', 'tb3')
        self.robot_type = self.get_parameter('robot_type').value
        qos_tb4 = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)

        if self.robot_type == 'tb4':
            self.cmd_topic           = '/tb4_0/cmd_vel'
            self.scan_topic          = '/tb4_0/scan'
            self.sub_qos             = qos_tb4
            self.front_index_offset  = 90
            self.use_intensity_filter = True
        elif self.robot_type == 'tb3':
            self.cmd_topic           = '/cmd_vel'
            self.scan_topic          = '/scan'
            self.sub_qos             = 10
            self.front_index_offset  = 0
            self.use_intensity_filter = False
        else:
            self.cmd_topic           = '/cmd_vel'
            self.scan_topic          = '/scan'
            self.sub_qos             = 10
            self.front_index_offset  = 0
            self.use_intensity_filter = False

        # Detección: obstáculo en el cono frontal ±window samples a < min_distance m
        self.min_distance = 0.30
        self.emergency_distance = 0.20
        self.window_deg = 12

        self.obstacle_detected = False

        # blocking: robot detenido, FSM en AVOIDING_OBSTACLE, planner replaneando.
        # cooldown: control devuelto, pausa antes de poder re-detectar.
        self.blocking_ticks = 0
        self.cooldown_ticks = 0
        self.BLOCKING_TICKS = 25  # 2.5 s — tiempo para que el planner genere nuevo path
        self.COOLDOWN_TICKS = 20  # 2.0 s — para que el robot arranque a girar/avanzar

        self.cmd_pub      = self.create_publisher(Twist, self.cmd_topic, 10)
        self.obstacle_pub = self.create_publisher(Bool, '/obstacle_detected', 10)
        self.scan_sub     = self.create_subscription(
            LaserScan, self.scan_topic, self.scan_callback, self.sub_qos
        )
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            f'Obstacle avoidance (stop-and-replan) ready — robot={self.robot_type}, '
            f'threshold={self.min_distance} m'
        )

    # ------------------------------------------------------------------
    def scan_callback(self, msg):
        n = len(msg.ranges)
        if n == 0:
            self.obstacle_detected = False
            return

        front_index = int(n * self.front_index_offset / 360)

        # ventana angular real, convertida a cantidad de índices según el scan
        samples_per_degree = n / 360.0
        window_indices = max(1, int(self.window_deg * samples_per_degree))

        valid = []

        for i in range(front_index - window_indices, front_index + window_indices + 1):
            idx = i % n
            r = msg.ranges[idx]

            if self.use_intensity_filter:
                if len(msg.intensities) <= idx or msg.intensities[idx] == 0.0:
                    continue

            if not math.isinf(r) and not math.isnan(r):
                valid.append(r)

        if not valid:
            self.obstacle_detected = False
            return

        min_r = min(valid)

        # emergencia: siempre detecta si está MUY cerca
        if min_r <= self.emergency_distance:
            self.obstacle_detected = True
            return

        self.obstacle_detected = min_r <= self.min_distance
    # ------------------------------------------------------------------
    def control_loop(self):
        obstacle_msg = Bool()

        # Fase 1: bloqueo activo — mantiene el robot frenado mientras el planner trabaja
        if self.blocking_ticks > 0:
            self.cmd_pub.publish(Twist())
            self.blocking_ticks -= 1
            obstacle_msg.data = True
            if self.blocking_ticks == 0:
                self.cooldown_ticks = self.COOLDOWN_TICKS
                self.get_logger().info('Blocking period over — releasing control to path_follower.')
            self.obstacle_pub.publish(obstacle_msg)
            return

        # Fase 2: cooldown — control ya devuelto, no re-detectar aún
        if self.cooldown_ticks > 0:
            self.cooldown_ticks -= 1
            obstacle_msg.data = False
            self.obstacle_pub.publish(obstacle_msg)
            return

        # Fase 3: detección normal
        if self.obstacle_detected:
            self.blocking_ticks = self.BLOCKING_TICKS
            self.cmd_pub.publish(Twist())   # frenar inmediatamente
            obstacle_msg.data = True
            self.get_logger().info(
                f'Obstacle detected (≤{self.min_distance} m) — stopping, triggering replan.'
            )
        else:
            obstacle_msg.data = False

        self.obstacle_pub.publish(obstacle_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
