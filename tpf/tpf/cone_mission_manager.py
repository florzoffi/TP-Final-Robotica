import math
import os
import time
from collections import deque
from enum import Enum

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import String, Bool
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
        self.inflation_radius_m = 0.18
        self.declare_parameter("exploration_waypoints", "")
        self.declare_parameter("snap_radius_cells", 8)
        self.declare_parameter("grid_spacing_m", 1)
        self.declare_parameter("coverage_radius_m", 1.5)
        self.declare_parameter("discard_radius_m", 0.6)

        raw_waypoints = self.get_parameter("exploration_waypoints").value
        self.waypoints = parse_waypoints(raw_waypoints)
        self.snap_radius_cells = self.get_parameter("snap_radius_cells").value
        self.grid_spacing_m = self.get_parameter("grid_spacing_m").value
        self.coverage_radius_m = self.get_parameter("coverage_radius_m").value
        self.discard_radius_m = self.get_parameter("discard_radius_m").value

        self.mission_state = MissionState.EXPLORING
        self.waypoint_index = 0
        self.current_pose = None

        self._pursuit_start_time = None
        self._pursuit_timeout_sec = 90.0
        self._pursuit_cone_goal = None   
        self._cone_arrival_radius_m = 0.25
        self._cone_goal_publish_time = None 
        self._discarded_cone_positions = []
        self._pending_cone_candidate = None
        self._raw_grid_cells = []  
        self._waypoints_ready = bool(self.waypoints)  
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

        self.plan_failed_sub = self.create_subscription(
            Bool,
            "/plan_failed",
            self.plan_failed_callback,
            10,
        )

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)

        self.declare_parameter("log_dir", os.path.expanduser("~/tpf_logs"))
        log_dir = os.path.expanduser(self.get_parameter("log_dir").value)
        os.makedirs(log_dir, exist_ok=True)

        csv_path = os.path.join(log_dir, "cone_detections.csv")
        self._csv = open(csv_path, "w")
        header = (
            "timestamp,cone_x,cone_y,distance_m,bearing_deg,"
            "goal_x,goal_y,path_found,path_length_m,num_waypoints,"
            "mission_outcome,pursuit_duration_sec\n"
        )
        self._csv.write(header)
        self._csv.flush()

        self._pending_row = None
        self._path_timeout_sec = 8.0
        self.create_timer(0.5, self._csv_timeout_check)
        self.create_timer(0.5, self._check_cone_arrival)
        self.create_timer(2.0, self._check_pursuit_timeout)

        fsm_log_path = os.path.join(log_dir, "mission_fsm_log.csv")
        self._fsm_log = open(fsm_log_path, "w")
        self._fsm_log.write("timestamp,source,from_state,to_state,trigger\n")
        self._fsm_log.flush()
        self._last_nav_state = None

        self.get_logger().info(
            f"Cone mission manager iniciado. Logs en: {log_dir} "
            f"(cone_detections.csv, mission_fsm_log.csv)"
        )

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
            if self.waypoints:
                self.start_exploration()   
            else:
                self._generate_raw_grid() 

    def start_exploration(self):
        if not self.waypoints:
            return
        self.mission_state = MissionState.EXPLORING
        self.waypoint_index = 0
        self._log_fsm_transition("INIT", "EXPLORING", "START")
        self.publish_waypoint_goal(self.waypoint_index)

    def nav_state_callback(self, msg):
        if msg.data != self._last_nav_state:
            self._log_fsm_transition(
                self._last_nav_state or "NONE", msg.data, "NAV_UPDATE", source="NAVIGATION"
            )
            self._last_nav_state = msg.data

        if (msg.data in ("LOCALIZED", "WAITING_GOAL")
                and not self._waypoints_ready
                and self.mission_state == MissionState.EXPLORING):
            self._apply_bfs_filter_and_start()
            return

        if msg.data != "GOAL_REACHED":
            return

        if self.mission_state == MissionState.PURSUING_CONE:
            if (self._cone_goal_publish_time is not None and
                    time.time() - self._cone_goal_publish_time < 2.0):
                self.get_logger().warn(
                    "GOAL_REACHED ignorado: llego muy rapido tras detectar cono "
                    "(probablemente era el waypoint anterior)."
                )
                return
            self.get_logger().info("Cono alcanzado (GOAL_REACHED) — mision completa.")
            self._complete_cone_pursuit("DONE", "GOAL_REACHED")
            return

        if self.mission_state == MissionState.EXPLORING:
            self.advance_exploration()

    def advance_exploration(self):
        if self.mission_state == MissionState.DONE:
            return

        self.waypoint_index += 1

        if self.waypoint_index >= len(self.waypoints):
            self.get_logger().info(
                "Exploracion completa — no quedan mas waypoints. Esperando deteccion de cono."
            )
            return

        self.publish_waypoint_goal(self.waypoint_index)

    def publish_waypoint_goal(self, index):
        if self.mission_state == MissionState.DONE:
            return
        x, y, yaw = self.waypoints[index]
        self.publish_goal(x, y, yaw)
        self.get_logger().info(f"Exploracion: publicando waypoint {index} -> ({x:.2f},{y:.2f})")

    def cone_callback(self, msg):
        if not self.map_received:
            self.get_logger().warn("Deteccion de cono recibida pero todavia no hay /map.")
            return

        raw_x = msg.point.x
        raw_y = msg.point.y

        if self._is_discarded_cone(raw_x, raw_y):
            self.get_logger().info(
                f"Deteccion en ({raw_x:.2f},{raw_y:.2f}) coincide con un cono ya "
                f"descartado en un intento previo — ignorando para no reprocesarlo."
            )
            return

        if self.mission_state != MissionState.EXPLORING:
            if self.mission_state == MissionState.PURSUING_CONE:
                self._remember_pending_cone(raw_x, raw_y)
            return

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
        self._pursue_cone(raw_x, raw_y, target_x, target_y)

    def _remember_pending_cone(self, x, y):
        if self._pursuit_cone_goal is not None:
            cx, cy = self._pursuit_cone_goal
            if math.hypot(x - cx, y - cy) <= self.discard_radius_m:
                return  
        if self._pending_cone_candidate is not None:
            px, py = self._pending_cone_candidate
            if math.hypot(x - px, y - py) <= self.discard_radius_m:
                return  
        self._pending_cone_candidate = (x, y)
        self.get_logger().info(
            f"Cono nuevo visto en ({x:.2f},{y:.2f}) mientras se persigue otro — "
            f"guardado como candidato pendiente."
        )

    def _pursue_cone(self, raw_x, raw_y, target_x, target_y):
        from_state = self.mission_state.value
        self._log_fsm_transition(from_state, "PURSUING_CONE", "CONE_DETECTED")
        self.mission_state = MissionState.PURSUING_CONE
        self._pursuit_start_time = time.time()
        self._cone_goal_publish_time = time.time()
        self._pursuit_cone_goal = (target_x, target_y)
        self.publish_goal(target_x, target_y, yaw=None)
        self._log_detection(raw_x, raw_y, target_x, target_y)

    def _stop_at_current_pose(self):
        """
        El goal del cono es su coordenada exacta, y el cono es un obstaculo
        fisico real: navigation_manager nunca puede satisfacer su tolerancia
        de posicion (0.12 m) contra el, asi que sin esto quedaria en un
        loop infinito PLANNING/FOLLOWING_PATH/AVOIDING_OBSTACLE (visto en
        mission_fsm_log.csv, 580+s de oscilacion tras un DONE).

        Al terminar la mision (sin cono pendiente) publicamos un goal en la
        pose actual del robot: navigation_manager lo da por satisfecho de
        inmediato y frena solo, sin que este nodo tenga que tocar /cmd_vel.
        """
        if self.current_pose is None:
            return
        x = self.current_pose.pose.position.x
        y = self.current_pose.pose.position.y
        q = self.current_pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z), 1.0 - 2.0 * (q.z ** 2))
        self.get_logger().info(
            f"Mision DONE — fijando goal en la pose actual ({x:.2f},{y:.2f}) "
            f"para que navigation_manager frene en vez de seguir "
            f"reintentando contra el cono."
        )
        self.publish_goal(x, y, yaw)

    def _start_pending_pursuit_or(self, fallback):
        """
        Si hay un cono pendiente guardado (visto mientras se perseguia otro),
        intenta perseguirlo ahora. Si no hay pendiente, o resulta invalido,
        ejecuta 'fallback' (retomar exploracion, o no hacer nada si la mision
        ya estaba DONE).
        """
        if self._pending_cone_candidate is not None:
            raw_x, raw_y = self._pending_cone_candidate
            self._pending_cone_candidate = None
            if not self._is_discarded_cone(raw_x, raw_y):
                target = self.validate_and_snap(raw_x, raw_y)
                if target is not None:
                    self.get_logger().info(
                        f"Retomando cono pendiente detectado durante la "
                        f"persecucion anterior: ({raw_x:.2f},{raw_y:.2f})."
                    )
                    self._pursue_cone(raw_x, raw_y, *target)
                    return
        fallback()

    def _generate_raw_grid(self):
        spacing = max(1, int(self.grid_spacing_m / self.map_resolution))
        candidates = []
        for r in range(0, self.map_height, spacing):
            for c in range(0, self.map_width, spacing):
                if self.is_free((r, c)):
                    candidates.append((r, c))
        self._raw_grid_cells = candidates
        self.get_logger().info(
            f"Grid de exploracion: {len(candidates)} candidatos "
            f"(espaciado {self.grid_spacing_m}m). Esperando pose inicial..."
        )

    def _apply_bfs_filter_and_start(self):
        if self.current_pose is None or not self._raw_grid_cells:
            return
        self._waypoints_ready = True 

        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        start_cell = self.world_to_map(rx, ry)
        if start_cell is None:
            self.get_logger().warn("Pose inicial fuera del mapa — no se puede generar waypoints.")
            return
        if not self.is_free(start_cell):
            start_cell = self.find_nearest_free_cell(start_cell)
        if start_cell is None:
            return

        self.get_logger().info("Calculando waypoints accesibles por BFS (puede tardar 1-2 s)...")
        reachable = self._bfs_reachable(start_cell)

        filtered = [c for c in self._raw_grid_cells if c in reachable]
        ordered = self._greedy_coverage_order(start_cell, filtered)

        self.waypoints = [
            (self.map_to_world(r, c)[0], self.map_to_world(r, c)[1], 0.0)
            for r, c in ordered
        ]
        self.get_logger().info(
            f"Waypoints de exploracion: {len(self.waypoints)} "
            f"(de {len(filtered)} accesibles / {len(self._raw_grid_cells)} candidatos) "
            f"tras podar por cobertura de vision."
        )
        self.start_exploration()

    def _bfs_reachable(self, start_cell):
        visited = {start_cell}
        queue = deque([start_cell])
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nb = (r + dr, c + dc)
                if nb not in visited and self.is_free(nb):
                    visited.add(nb)
                    queue.append(nb)
        return visited

    def _greedy_coverage_order(self, start, cells):
        """
        Ordena los waypoints con un greedy tipo set-cover: en cada paso elige
        el candidato que maximiza area NUEVA cubierta (dentro de
        coverage_radius_m y con linea de vista libre, para no "ver" a traves
        de paredes) por unidad de distancia, en vez de solo el mas cercano.

        Una vez que ningun candidato restante aporta cobertura nueva (todo lo
        que queda ya cae dentro del area vista por waypoints previos), corta
        el recorrido ahi: agregar esos puntos no mejora la busqueda visual de
        conos, solo suma distancia recorrida.
        """
        if not cells:
            return []

        radius_cells = self.coverage_radius_m / self.map_resolution
        remaining = list(cells)
        covered = set()
        ordered = []
        current = start

        while remaining:
            best_cell = None
            best_gain = -1
            best_dist = None
            for c in remaining:
                dist = math.hypot(c[0] - current[0], c[1] - current[1])
                gain = self._coverage_gain(c, remaining, covered, radius_cells)
                if gain > best_gain or (gain == best_gain and (best_dist is None or dist < best_dist)):
                    best_gain = gain
                    best_dist = dist
                    best_cell = c

            if best_gain <= 0 and ordered:
                break

            ordered.append(best_cell)
            remaining.remove(best_cell)
            covered.add(best_cell)
            for c in remaining:
                if c not in covered \
                        and math.hypot(c[0] - best_cell[0], c[1] - best_cell[1]) <= radius_cells \
                        and self._line_of_sight(best_cell, c):
                    covered.add(c)
            current = best_cell

        return ordered

    def _coverage_gain(self, center, remaining, covered, radius_cells):
        gain = 0
        for c in remaining:
            if c in covered:
                continue
            if math.hypot(c[0] - center[0], c[1] - center[1]) > radius_cells:
                continue
            if not self._line_of_sight(center, c):
                continue
            gain += 1
        return gain

    def _line_of_sight(self, cell_a, cell_b):
        """
        Bresenham entre dos celdas de grilla sobre el occupancy grid crudo
        (no el inflado: el inflado es para margen de navegacion, no para
        vision). Si alguna celda intermedia esta ocupada o es desconocida,
        se considera que no hay linea de vista directa entre ambas.
        """
        r0, c0 = cell_a
        r1, c1 = cell_b
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc
        r, c = r0, c0

        while (r, c) != (r1, c1):
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc
            if (r, c) == (r1, c1):
                break
            idx = r * self.map_width + c
            occ = self.occupancy_data[idx]
            if occ > 50 or occ == -1:
                return False
        return True

    def _dist_to_cone_goal(self):
        """Distancia actual del robot al goal del cono (inf si falta info)."""
        if self.current_pose is None or self._pursuit_cone_goal is None:
            return float('inf')
        rx = self.current_pose.pose.position.x
        ry = self.current_pose.pose.position.y
        cx, cy = self._pursuit_cone_goal
        return math.hypot(rx - cx, ry - cy)

    def _complete_cone_pursuit(self, outcome, trigger):
        """
        Cierra la persecucion actual como exito (DONE): escribe el CSV, loguea
        la transicion, resetea el estado de persecucion y encadena un cono
        pendiente si lo hay (o frena en la pose actual). Centraliza lo que
        antes estaba duplicado entre GOAL_REACHED y el chequeo de cercania.
        """
        duration = (time.time() - self._pursuit_start_time
                    if self._pursuit_start_time else 0.0)
        if self._pending_row is not None:
            self._write_csv_row(self._pending_row, outcome=outcome,
                                pursuit_duration_sec=duration)
            self._pending_row = None
        self._log_fsm_transition("PURSUING_CONE", "DONE", trigger)
        self.mission_state = MissionState.DONE
        self._pursuit_start_time = None
        self._pursuit_cone_goal = None
        self._cone_goal_publish_time = None
        self._start_pending_pursuit_or(self._stop_at_current_pose)

    def _check_cone_arrival(self):
        """
        Corre continuamente (cada 0.5s): si el robot entro al radio de arribo
        del cono, da la mision por completada SIN esperar el timeout de 90s.

        Motivo: obstacle_avoidance frena el robot a ~0.3m del cono (que es un
        obstaculo fisico), asi que path_follower nunca llega a reportar
        GOAL_REACHED contra el centro del cono y entra en bucle
        FOLLOWING_PATH/AVOIDING_OBSTACLE. Con este chequeo, apenas el robot
        esta lo bastante cerca, se considera la aproximacion exitosa.
        """
        if self.mission_state != MissionState.PURSUING_CONE:
            return
        if self._pursuit_start_time is None:
            return

        if (self._cone_goal_publish_time is not None and
                time.time() - self._cone_goal_publish_time < 2.0):
            return
        if self._dist_to_cone_goal() <= self._cone_arrival_radius_m:
            self.get_logger().info(
                f"Robot a {self._dist_to_cone_goal():.2f}m del cono "
                f"(<= {self._cone_arrival_radius_m:.2f}m) — aproximacion completa."
            )
            self._complete_cone_pursuit("DONE_NEAR", "ARRIVAL_NEAR")

    def _check_pursuit_timeout(self):
        if self.mission_state != MissionState.PURSUING_CONE:
            return
        if self._pursuit_start_time is None:
            return
        if time.time() - self._pursuit_start_time <= self._pursuit_timeout_sec:
            return

        dist_to_cone = self._dist_to_cone_goal()

        if dist_to_cone <= self._cone_arrival_radius_m:
            self.get_logger().info(
                f"Timeout pero robot a {dist_to_cone:.2f}m del cono — mision completa."
            )
            self._complete_cone_pursuit("DONE_NEAR_TIMEOUT", "TIMEOUT_NEAR")
        else:
            self.get_logger().warn(
                f"Timeout de {self._pursuit_timeout_sec:.0f}s persiguiendo cono "
                f"(dist={dist_to_cone:.2f}m) — volviendo a explorar."
            )
            duration = time.time() - self._pursuit_start_time
            if self._pending_row is not None:
                self._write_csv_row(self._pending_row, outcome="TIMEOUT",
                                    pursuit_duration_sec=duration)
            self._abandon_cone_pursuit("TIMEOUT_FAR")
            self._pending_row = None
            self._start_pending_pursuit_or(self.advance_exploration)

    def _is_discarded_cone(self, x, y):
        return any(
            math.hypot(x - dx, y - dy) <= self.discard_radius_m
            for dx, dy in self._discarded_cone_positions
        )

    def _abandon_cone_pursuit(self, trigger):
        """
        Corta la persecucion del cono actual y vuelve a EXPLORING, guardando
        su coordenada cruda como descartada para no volver a perseguirlo si
        se re-detecta mientras se sigue explorando el laberinto.
        """
        if self._pending_row is not None:
            self._discarded_cone_positions.append(
                (self._pending_row["cone_x"], self._pending_row["cone_y"])
            )
        self._log_fsm_transition("PURSUING_CONE", "EXPLORING", trigger)
        self.mission_state = MissionState.EXPLORING
        self._pursuit_start_time = None
        self._pursuit_cone_goal = None
        self._cone_goal_publish_time = None

    def _log_fsm_transition(self, from_state, to_state, trigger, source="MISSION"):
        ts = time.time()
        self._fsm_log.write(f"{ts:.3f},{source},{from_state},{to_state},{trigger}\n")
        self._fsm_log.flush()

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
            "path_found": False,
            "path_length_m": 0.0,
            "num_waypoints": 0,
        }

    def plan_callback(self, msg):
        if self._pending_row is None or len(msg.poses) == 0:
            return
        if self._pending_row.get("path_found"):
            return 
        length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i - 1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i - 1].pose.position.y
            length += math.hypot(dx, dy)
        self._pending_row["path_found"] = True
        self._pending_row["path_length_m"] = length
        self._pending_row["num_waypoints"] = len(msg.poses)

    def plan_failed_callback(self, msg):
        if self._pending_row is None or self._pending_row.get("path_found"):
            return
        elapsed = time.time() - self._pending_row["timestamp"]
        self._declare_no_path(elapsed)

    def _csv_timeout_check(self):
        if self._pending_row is None:
            return
        if self._pending_row.get("path_found"):
            return
        elapsed = time.time() - self._pending_row["timestamp"]
        if elapsed > self._path_timeout_sec:
            self._declare_no_path(elapsed)

    def _declare_no_path(self, elapsed):
        self._write_csv_row(self._pending_row, outcome="NO_PATH",
                            pursuit_duration_sec=elapsed)
        if self.mission_state == MissionState.PURSUING_CONE:
            self._abandon_cone_pursuit("NO_PATH")
            self._start_pending_pursuit_or(self.advance_exploration)
        self._pending_row = None

    def _write_csv_row(self, row, outcome, pursuit_duration_sec=0.0):
        self._csv.write(
            f"{row['timestamp']:.3f},"
            f"{row['cone_x']:.4f},{row['cone_y']:.4f},"
            f"{row['distance_m']:.4f},{row['bearing_deg']:.2f},"
            f"{row['goal_x']:.4f},{row['goal_y']:.4f},"
            f"{'True' if row.get('path_found') else 'False'},"
            f"{row.get('path_length_m', 0.0):.4f},{row.get('num_waypoints', 0)},"
            f"{outcome},{pursuit_duration_sec:.1f}\n"
        )
        self._csv.flush()
        self.get_logger().info(
            f"CSV: cono=({row['cone_x']:.2f},{row['cone_y']:.2f}) "
            f"dist={row['distance_m']:.2f}m bearing={row['bearing_deg']:.1f}deg "
            f"path={row.get('path_length_m', 0.0):.2f}m "
            f"outcome={outcome} duracion={pursuit_duration_sec:.1f}s"
        )

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

        if self.current_pose is not None:
            rx = self.current_pose.pose.position.x
            ry = self.current_pose.pose.position.y
            start_cell = self.world_to_map(rx, ry)
        else:
            start_cell = None

        if start_cell is None or not self.is_free(start_cell):
            if self.is_free(goal_cell):
                return x, y
            nearest = self.find_nearest_free_cell(goal_cell, max_radius=self.snap_radius_cells)
            return self.map_to_world(*nearest) if nearest else None
        
        BFS_RADIUS = 60
        gr, gc = goal_cell
        sr, sc = start_cell

        visited = {start_cell}
        queue = deque([start_cell])
        best_cell = start_cell
        best_dist = math.hypot(sr - gr, sc - gc)

        while queue:
            r, c = queue.popleft()
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
    node._fsm_log.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
