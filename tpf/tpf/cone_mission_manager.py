import math
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid


class MissionState(Enum):
    EXPLORING = "EXPLORING"
    PURSUING_CONE = "PURSUING_CONE"
    DONE = "DONE"


def parse_waypoints(raw):
    """
    Parsea "x0,y0,yaw0;x1,y1,yaw1;..." en una lista de tuplas (x, y, yaw).
    Los parametros ROS2 no soportan bien arrays anidados, asi que la lista
    de waypoints se pasa como un solo string y se "reshapea" aca.
    """
    waypoints = []
    raw = raw.strip()
    if not raw:
        return waypoints

    for group in raw.split(";"):
        group = group.strip()
        if not group:
            continue
        parts = [p.strip() for p in group.split(",")]
        if len(parts) != 3:
            continue
        x, y, yaw = (float(p) for p in parts)
        waypoints.append((x, y, yaw))

    return waypoints


class ConeMissionManager(Node):
    """
    FSM de mision de alto nivel: explora el mapa real publicando waypoints
    en /goal_pose, y si recibe una deteccion confirmada de cono en
    /cone_detection, interrumpe la exploracion y navega hacia el cono.

    Nunca publica /cmd_vel ni mueve el robot directamente — solo publica
    /goal_pose, exactamente igual que cualquier otro originador de goals
    en este proyecto. Toda la evasion de paredes (incluyendo el caso de
    "vi el cono a traves de un hueco") la resuelve el Theta* que ya existe
    en path_planner.py, una vez que la coordenada publicada cae en una
    celda libre del grid inflado.
    """

    def __init__(self):
        super().__init__("cone_mission_manager")

        self.inflation_radius_m = 0.25

        self.declare_parameter("exploration_waypoints", "")
        self.declare_parameter("snap_radius_cells", 8)

        raw_waypoints = self.get_parameter("exploration_waypoints").value
        self.waypoints = parse_waypoints(raw_waypoints)
        self.snap_radius_cells = self.get_parameter("snap_radius_cells").value

        if not self.waypoints:
            self.get_logger().warn(
                "exploration_waypoints vacio — no se publicara ningun waypoint de "
                "exploracion hasta que se setee el parametro (tipicamente derivado "
                "del mapa real construido en la Fase 1)."
            )

        self.mission_state = MissionState.EXPLORING
        self.waypoint_index = 0

        # ---------------- estado del mapa (replica path_planner.py) ----------------
        self.map_received = False
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.occupancy_data = None
        self.inflated_grid = None

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            "/map",
            self.map_callback,
            map_qos,
        )

        self.nav_state_sub = self.create_subscription(
            String,
            "/navigation_state",
            self.nav_state_callback,
            10,
        )

        self.cone_sub = self.create_subscription(
            PointStamped,
            "/cone_detection",
            self.cone_callback,
            10,
        )

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.get_logger().info("Cone mission manager iniciado. Esperando /map...")

    # ------------------------------------------------------------------
    def map_callback(self, msg):
        already_had_map = self.map_received

        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.occupancy_data = list(msg.data)

        self.build_inflated_grid()
        self.map_received = True

        if not already_had_map:
            self.get_logger().info(
                f"Mapa recibido: {self.map_width}x{self.map_height}, "
                f"res={self.map_resolution:.3f}"
            )
            self.start_exploration()

    def start_exploration(self):
        if not self.waypoints:
            return
        self.mission_state = MissionState.EXPLORING
        self.waypoint_index = 0
        self.publish_waypoint_goal(self.waypoint_index)

    # ------------------------------------------------------------------
    def nav_state_callback(self, msg):
        if msg.data != "GOAL_REACHED":
            return

        if self.mission_state == MissionState.PURSUING_CONE:
            self.get_logger().info("Cono alcanzado — mision completa.")
            self.mission_state = MissionState.DONE
            return

        if self.mission_state == MissionState.EXPLORING:
            self.advance_exploration()

    def advance_exploration(self):
        self.waypoint_index += 1

        if self.waypoint_index >= len(self.waypoints):
            self.get_logger().info(
                "Exploracion completa — no quedan mas waypoints. Esperando deteccion de cono."
            )
            return

        self.publish_waypoint_goal(self.waypoint_index)

    def publish_waypoint_goal(self, index):
        x, y, yaw = self.waypoints[index]
        self.publish_goal(x, y, yaw)
        self.get_logger().info(f"Exploracion: publicando waypoint {index} -> ({x:.2f},{y:.2f})")

    # ------------------------------------------------------------------
    def cone_callback(self, msg):
        if self.mission_state != MissionState.EXPLORING:
            # Ya estamos persiguiendo un cono o la mision termino — ignorar
            # detecciones adicionales (multi-cono queda fuera de alcance).
            return

        if not self.map_received:
            self.get_logger().warn("Deteccion de cono recibida pero todavia no hay /map.")
            return

        raw_x = msg.point.x
        raw_y = msg.point.y

        target = self.validate_and_snap(raw_x, raw_y)
        if target is None:
            self.get_logger().warn(
                f"Deteccion de cono en ({raw_x:.2f},{raw_y:.2f}) cae en pared/desconocido "
                f"y no se encontro celda libre cercana — descartando deteccion."
            )
            return

        target_x, target_y = target
        if (target_x, target_y) != (raw_x, raw_y):
            self.get_logger().info(
                f"Cono visto en zona ocupada — ajustando objetivo de "
                f"({raw_x:.2f},{raw_y:.2f}) a celda libre ({target_x:.2f},{target_y:.2f})."
            )

        self.get_logger().info(
            f"Interrumpiendo exploracion — persiguiendo cono en ({target_x:.2f},{target_y:.2f})."
        )

        self.mission_state = MissionState.PURSUING_CONE
        self.publish_goal(target_x, target_y, yaw=None)

    def validate_and_snap(self, x, y):
        """
        Si (x, y) cae en una celda libre del grid inflado, se devuelve tal
        cual. Si no (el caso de "vi el cono a traves de un hueco/reja" —
        la celda reportada esta dentro o pegada a una pared), se busca la
        celda libre mas cercana dentro de un radio chico y se devuelve esa
        en su lugar. Si no hay ninguna celda libre cerca, se devuelve None.
        """
        cell = self.world_to_map(x, y)
        if cell is None:
            return None

        if self.is_free(cell):
            return x, y

        nearest = self.find_nearest_free_cell(cell, max_radius=self.snap_radius_cells)
        if nearest is None:
            return None

        return self.map_to_world(*nearest)

    def publish_goal(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0

        if yaw is None:
            msg.pose.orientation.w = 1.0
        else:
            msg.pose.orientation.z = math.sin(yaw / 2.0)
            msg.pose.orientation.w = math.cos(yaw / 2.0)

        self.goal_pub.publish(msg)

    # ------------------------------------------------------------------
    # Replica de path_planner.py: misma logica de inflado / busqueda de
    # celda libre, para garantizar que cualquier goal que publiquemos sea
    # aceptable por el planner (que usa exactamente esta misma definicion
    # de "libre").
    # ------------------------------------------------------------------
    def build_inflated_grid(self):
        width = self.map_width
        height = self.map_height
        res = self.map_resolution

        inflation_cells = int(math.ceil(self.inflation_radius_m / res))

        self.inflated_grid = [[0 for _ in range(width)] for _ in range(height)]

        occupied_cells = []
        for row in range(height):
            for col in range(width):
                idx = row * width + col
                occ = self.occupancy_data[idx]
                if occ > 50 or occ == -1:
                    occupied_cells.append((row, col))

        for row, col in occupied_cells:
            for dr in range(-inflation_cells, inflation_cells + 1):
                for dc in range(-inflation_cells, inflation_cells + 1):
                    nr = row + dr
                    nc = col + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    dist = math.sqrt(dr * dr + dc * dc) * res
                    if dist <= self.inflation_radius_m:
                        self.inflated_grid[nr][nc] = 1

    def is_free(self, cell):
        row, col = cell
        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return False
        return self.inflated_grid[row][col] == 0

    def find_nearest_free_cell(self, cell, max_radius=8):
        row, col = cell
        for r in range(1, max_radius + 1):
            for dr in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if abs(dr) != r and abs(dc) != r:
                        continue
                    candidate = (row + dr, col + dc)
                    if self.is_free(candidate):
                        return candidate
        return None

    def world_to_map(self, x, y):
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)
        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None
        return row, col

    def map_to_world(self, row, col):
        x = self.map_origin_x + (col + 0.5) * self.map_resolution
        y = self.map_origin_y + (row + 0.5) * self.map_resolution
        return x, y


def main(args=None):
    rclpy.init(args=args)

    node = ConeMissionManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
