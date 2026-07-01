import math
import os
import time
from collections import deque
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Path


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
        self.current_pose = None

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

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/estimated_pose",
            lambda msg: setattr(self, "current_pose", msg),
            10,
        )

        self.plan_sub = self.create_subscription(
            Path,
            "/plan",
            self.plan_callback,
            10,
        )

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        # ---------------- CSV de detecciones ----------------
        csv_path = "src/TP-Final-Robotica/tpf/cone_detections.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self._csv = open(csv_path, "w")
        header = "timestamp,cone_x,cone_y,distance_m,bearing_deg,goal_x,goal_y,path_found,path_length_m,num_waypoints\n"
        self._csv.write(header)
        self._csv.flush()
        self._pending_row = None   # fila esperando saber si se encontro camino
        self._path_timeout_sec = 4.0
        self.create_timer(0.5, self._csv_timeout_check)

        self.get_logger().info(f"Cone mission manager iniciado. CSV: {csv_path}")

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
        self._log_detection(raw_x, raw_y, target_x, target_y)

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------
    def _log_detection(self, cone_x, cone_y, goal_x, goal_y):
        ts = time.time()
        distance, bearing = 0.0, 0.0
        if self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
            siny = 2.0 * (self.current_pose.pose.orientation.w * self.current_pose.pose.orientation.z)
            cosy = 1.0 - 2.0 * self.current_pose.pose.orientation.z ** 2
            yaw = math.atan2(siny, cosy)
            distance = math.hypot(cone_x - rx, cone_y - ry)
            bearing = math.degrees(math.atan2(cone_y - ry, cone_x - rx) - yaw)

        self._pending_row = {
            "timestamp": ts,
            "cone_x": cone_x,
            "cone_y": cone_y,
            "distance_m": distance,
            "bearing_deg": bearing,
            "goal_x": goal_x,
            "goal_y": goal_y,
        }

    def plan_callback(self, msg):
        if self._pending_row is None or len(msg.poses) == 0:
            return
        length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i - 1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i - 1].pose.position.y
            length += math.hypot(dx, dy)
        self._write_csv_row(self._pending_row, path_found=True,
                            path_length_m=length, num_waypoints=len(msg.poses))
        self._pending_row = None

    def _csv_timeout_check(self):
        if self._pending_row is None:
            return
        elapsed = time.time() - self._pending_row["timestamp"]
        if elapsed > self._path_timeout_sec:
            self._write_csv_row(self._pending_row, path_found=False,
                                path_length_m=0.0, num_waypoints=0)
            self._pending_row = None

    def _write_csv_row(self, row, path_found, path_length_m=0.0, num_waypoints=0):
        self._csv.write(
            f"{row['timestamp']:.3f},"
            f"{row['cone_x']:.4f},{row['cone_y']:.4f},"
            f"{row['distance_m']:.4f},{row['bearing_deg']:.2f},"
            f"{row['goal_x']:.4f},{row['goal_y']:.4f},"
            f"{'True' if path_found else 'False'},"
            f"{path_length_m:.4f},{num_waypoints}\n"
        )
        self._csv.flush()
        self.get_logger().info(
            f"CSV: cono=({row['cone_x']:.2f},{row['cone_y']:.2f}) "
            f"dist={row['distance_m']:.2f}m bearing={row['bearing_deg']:.1f}deg "
            f"goal=({row['goal_x']:.2f},{row['goal_y']:.2f}) "
            f"path_found={path_found} length={path_length_m:.2f}m waypoints={num_waypoints}"
        )

    # ------------------------------------------------------------------
    def validate_and_snap(self, x, y):
        """
        Encuentra la celda libre MAS CERCANA AL CONO que sea ALCANZABLE
        desde la posicion actual del robot (BFS desde start_cell).

        El snap simple (celda libre mas cercana al cono) falla cuando el cono
        esta en una bolsa de espacio libre desconectada de donde esta el robot
        — exactamente el caso de "vi el cono a traves de un hueco en la pared".
        El BFS garantiza que el goal publicado sea alcanzable por Theta*.
        """
        goal_cell = self.world_to_map(x, y)
        if goal_cell is None:
            return None

        # Obtener celda de inicio desde la pose estimada del robot
        if self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
            start_cell = self.world_to_map(rx, ry)
        else:
            start_cell = None

        if start_cell is None or not self.is_free(start_cell):
            # Sin pose del robot: fallback al snap geometrico simple
            if self.is_free(goal_cell):
                return x, y
            nearest = self.find_nearest_free_cell(goal_cell, max_radius=self.snap_radius_cells)
            return self.map_to_world(*nearest) if nearest else None

        # BFS desde el robot: explora todas las celdas libres accesibles
        # y devuelve la mas cercana (Euclideana de celda) al cono.
        # Limita la exploracion a BFS_RADIUS celdas desde el robot para
        # no recorrer el mapa entero (a 0.05 m/celda, 150 celdas = 7.5 m).
        # 60 celdas * 0.05 m/celda = 3 m de radio — mas que suficiente para
        # encontrar el punto de aproximacion, y pequeno para que el BFS en
        # Python termine en milisegundos sin bloquear el callback de ROS.
        BFS_RADIUS = 60
        gr, gc = goal_cell
        sr, sc = start_cell

        visited = {start_cell}
        queue = deque([start_cell])
        best_cell = start_cell
        best_dist = math.hypot(sr - gr, sc - gc)

        while queue:
            r, c = queue.popleft()
            # OR: cortar si supera el radio en CUALQUIER direccion (no ambas).
            if abs(r - sr) > BFS_RADIUS or abs(c - sc) > BFS_RADIUS:
                continue

            d = math.hypot(r - gr, c - gc)
            if d < best_dist:
                best_dist = d
                best_cell = (r, c)

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nb = (r + dr, c + dc)
                if nb not in visited and self.is_free(nb):
                    visited.add(nb)
                    queue.append(nb)

        best_world = self.map_to_world(*best_cell)

        if best_dist > 2.0 / self.map_resolution:
            # La celda alcanzable mas cercana esta a mas de 2 m del cono:
            # el cono es inaccesible desde esta posicion, no vale la pena
            # intentarlo. Descartamos la deteccion para no mandar al robot
            # a un punto arbitrario lejos del cono.
            self.get_logger().warn(
                f"Cono en ({x:.2f},{y:.2f}) inaccesible desde robot — "
                f"mejor celda alcanzable a {best_dist * self.map_resolution:.1f}m del cono. "
                f"Descartando deteccion."
            )
            return None

        if best_cell != goal_cell:
            self.get_logger().info(
                f"BFS snap: cono en ({x:.2f},{y:.2f}) -> "
                f"celda alcanzable mas cercana ({best_world[0]:.2f},{best_world[1]:.2f}) "
                f"a {best_dist * self.map_resolution:.2f}m del cono."
            )

        return best_world

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

    node._csv.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
