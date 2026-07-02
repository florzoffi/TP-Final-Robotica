import math
import heapq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan



class PathPlanner(Node):

    def __init__(self):
        super().__init__("path_planner")

        # Radio de seguridad alrededor de paredes onda para no acercarnos
        # El robot no es un punto, entonces inflamos obstáculos.
        self.inflation_radius_m = 0.25

        self.map_received = False
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.occupancy_data = None
        self.inflated_grid = None

        self.current_pose = None
        self.goal_pose = None

        self.last_plan_pose = None
        self.replan_distance_threshold = 0.30

        # Radio de inflado para obstáculos dinámicos (más pequeño que el estático
        # para no bloquear corredores estrechos).
        self.dynamic_inflation_radius_m = 0.18

        # Memoria de obstáculos dinámicos: (row, col) -> Time de la última
        # detección. Un obstáculo detectado queda "recordado" durante
        # DYNAMIC_DECAY_SEC aunque el LIDAR deje de verlo en el scan actual
        # (p.ej. porque el robot giró). Sin esto, el planner replanifica
        # contra un cluster de obstáculos distinto cada vez que ve un
        # pedazo distinto del mismo cluster, y el camino elegido cambia de
        # lado constantemente.
        self.dynamic_obstacle_points = {}
        self.DYNAMIC_DECAY_SEC = 30.0
        self.consecutive_static_fallbacks = 0
        self.MAX_CONSECUTIVE_FALLBACKS = 3

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

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/estimated_pose",
            self.pose_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10,
        )

        self.path_pub = self.create_publisher(
            Path,
            "/plan",
            10,
        )
        
        #aca basicamente lo que hago es replanificar despues de un ratio 
        #como para que si la pose estimada cambia, se actaulice el path
        self.replan_timer = self.create_timer(
            1.0,
            self.replan_timer_callback,
        )

        self.get_logger().info("Path planner iniciado. Esperando /map, /estimated_pose y /goal_pose...")

    def map_callback(self, msg):
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.occupancy_data = list(msg.data)

        self.build_inflated_grid()
        self.dynamic_obstacle_points = {}

        self.map_received = True

        self.get_logger().info(
            f"Mapa recibido para planificacion: {self.map_width}x{self.map_height}, "
            f"res={self.map_resolution:.3f}"
        )

    def pose_callback(self, msg):
        self.current_pose = msg

    def scan_callback(self, msg):
        self.register_dynamic_obstacles(msg)

    def goal_callback(self, msg):
        self.goal_pose = msg
        self.last_plan_pose = None
        self.get_logger().info("Nuevo goal recibido. Intentando planificar...")
        self.plan_if_possible()

    def replan_timer_callback(self):
        if self.goal_pose is None:
            return

        if self.current_pose is None:
            return

        if not self.map_received:
            return

        if self.last_plan_pose is None:
            self.plan_if_possible()
            return

        current_x = self.current_pose.pose.position.x
        current_y = self.current_pose.pose.position.y

        last_x, last_y = self.last_plan_pose

        dx = current_x - last_x
        dy = current_y - last_y

        distance_from_last_plan = math.sqrt(dx * dx + dy * dy)

        if distance_from_last_plan > self.replan_distance_threshold:
            self.get_logger().info(
                f"Replanificando: robot se movio {distance_from_last_plan:.2f} m desde el ultimo plan."
            )
            self.plan_if_possible()
        

    def build_inflated_grid(self):
        """
        Crea una grilla de planificación.
        0 = libre
        1 = ocupado o demasiado cerca de una pared
        """
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

                # Ocupado o desconocido.
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

    def find_nearest_free_cell(self, cell, max_radius=8):
        """
        Busca la celda libre más cercana a cell en un radio creciente.
        Útil cuando la pose estimada cae en una zona inflada tras una rotación.
        """
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

    def is_free(self, cell):
        row, col = cell

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return False

        return self.inflated_grid[row][col] == 0

    # ============================================================
    # CONVERSION MUNDO <-> MAPA
    # ============================================================

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

    # ============================================================
    # PLANIFICACION
    # ============================================================

    def yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def register_dynamic_obstacles(self, scan):
        """
        Procesa un scan de LIDAR y actualiza la memoria de obstáculos
        dinámicos (self.dynamic_obstacle_points) con marca de tiempo.

        Solo se registran lecturas dentro de MAX_DYNAMIC_RANGE metros, y
        solo si la celda no está ya ocupada en el mapa estático (para no
        "redescubrir" paredes ya conocidas como si fueran dinámicas).
        """
        if self.current_pose is None or self.inflated_grid is None:
            return

        MAX_DYNAMIC_RANGE = 1.0  # m — casi el rango completo del LIDAR del TB3
        MAX_DYNAMIC_BEARING = math.radians(45)
        
        robot_x   = self.current_pose.pose.position.x
        robot_y   = self.current_pose.pose.position.y
        robot_yaw = self.yaw_from_quaternion(self.current_pose.pose.orientation)

        now = self.get_clock().now()

        for i, r in enumerate(scan.ranges):
            if math.isinf(r) or math.isnan(r) or r > MAX_DYNAMIC_RANGE:
                continue

            angle     = scan.angle_min + i * scan.angle_increment
            if abs(angle) > MAX_DYNAMIC_BEARING:
                continue
            world_ang = robot_yaw + angle
            obs_x     = robot_x + r * math.cos(world_ang)
            obs_y     = robot_y + r * math.sin(world_ang)

            obs_cell = self.world_to_map(obs_x, obs_y)
            if obs_cell is None:
                continue

            obs_row, obs_col = obs_cell

            # Solo marcamos como dinámico si el centro NO estaba ya
            # bloqueado por el mapa estático. Así evitamos "redescubrir"
            # paredes ya conocidas como si fueran obstáculos dinámicos.
            if self.inflated_grid[obs_row][obs_col] != 0:
                continue

            self.dynamic_obstacle_points[(obs_row, obs_col)] = now

    def build_dynamic_grid(self):
        """
        Devuelve una copia del mapa inflado estático con los obstáculos
        dinámicos "recordados" incorporados.

        Usa self.dynamic_obstacle_points en vez del último scan únicamente:
        cada punto detectado queda vivo durante DYNAMIC_DECAY_SEC, así que
        un obstáculo no desaparece de la grilla solo porque el robot giró y
        el LIDAR dejó de verlo por un instante. Esto evita que Theta* elija
        un lado distinto del mismo cluster en cada replanificación.
        """
        # Copia superficial fila a fila (las filas son listas independientes).
        dynamic = [row[:] for row in self.inflated_grid]

        if not self.dynamic_obstacle_points:
            return dynamic

        now = self.get_clock().now()
        inf_cells = int(math.ceil(self.dynamic_inflation_radius_m / self.map_resolution))

        expired = []

        for (obs_row, obs_col), last_seen in self.dynamic_obstacle_points.items():
            age_sec = (now - last_seen).nanoseconds / 1e9

            if age_sec > self.DYNAMIC_DECAY_SEC:
                expired.append((obs_row, obs_col))
                continue
            
            
            for dr in range(-inf_cells, inf_cells + 1):
                for dc in range(-inf_cells, inf_cells + 1):
                    if math.sqrt(dr * dr + dc * dc) * self.map_resolution <= self.dynamic_inflation_radius_m:
                        nr = obs_row + dr
                        nc = obs_col + dc
                        if 0 <= nr < self.map_height and 0 <= nc < self.map_width:
                            dynamic[nr][nc] = 1

        for key in expired:
            del self.dynamic_obstacle_points[key]

        return dynamic

    def plan_if_possible(self):
        if not self.map_received:
            self.get_logger().warn("No puedo planificar: todavia no llego /map.")
            return

        if self.current_pose is None:
            self.get_logger().warn("No puedo planificar: todavia no llego /estimated_pose.")
            return

        if self.goal_pose is None:
            self.get_logger().warn("No puedo planificar: todavia no llego /goal_pose.")
            return

        start_x = self.current_pose.pose.position.x
        start_y = self.current_pose.pose.position.y

        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        start_cell = self.world_to_map(start_x, start_y)
        goal_cell = self.world_to_map(goal_x, goal_y)

        if start_cell is None:
            self.get_logger().warn("La pose inicial cae fuera del mapa.")
            return

        if goal_cell is None:
            self.get_logger().warn("El goal cae fuera del mapa.")
            return

        if not self.is_free(start_cell):
            start_cell = self.find_nearest_free_cell(start_cell)
            if start_cell is None:
                self.get_logger().warn("La celda inicial esta ocupada o demasiado cerca de una pared.")
                return
            self.get_logger().info("Celda inicial ocupada — usando celda libre cercana.")

        if not self.is_free(goal_cell):
            self.get_logger().warn(
                f"DIAG planner: goal ({goal_x:.2f},{goal_y:.2f}) -> celda {goal_cell} OCUPADA."
            )
            return

        # DIAG: loguear start/goal en coordenadas de mundo Y de celda para
        # poder ubicarlos visualmente en el mapa y entender por que Theta* falla.
        start_world = self.map_to_world(*start_cell)
        goal_world = self.map_to_world(*goal_cell)
        self.get_logger().info(
            f"DIAG planner: start mundo=({start_x:.2f},{start_y:.2f}) "
            f"celda={start_cell} libre={self.is_free(start_cell)} | "
            f"goal mundo=({goal_x:.2f},{goal_y:.2f}) -> "
            f"celda={goal_cell} libre={self.is_free(goal_cell)} | "
            f"start_snap=({start_world[0]:.2f},{start_world[1]:.2f}) "
            f"goal_snap=({goal_world[0]:.2f},{goal_world[1]:.2f})",
            throttle_duration_sec=2.0,
        )

        # Planifica sobre mapa estático + obstáculos dinámicos del LIDAR.
        static_grid = self.inflated_grid
        self.inflated_grid = self.build_dynamic_grid()

        path_cells = self.theta_star(start_cell, goal_cell)

        self.inflated_grid = static_grid  # restaurar siempre

        if path_cells is None:
            self.consecutive_static_fallbacks += 1

            self.get_logger().warn(
                "No se encontro camino con obstaculos dinamicos — "
                "reintentando solo con mapa estatico."
            )

            if self.consecutive_static_fallbacks >= self.MAX_CONSECUTIVE_FALLBACKS:
                self.get_logger().error(
                    f"Fallback estatico repetido {self.consecutive_static_fallbacks} veces. "
                    "Probable obstaculo dinamico bloqueando el paso o memoria dinamica deformando el mapa."
                )

            path_cells = self.theta_star(start_cell, goal_cell)

            if path_cells is None:
                self.get_logger().warn(
                    "No se encontro camino ni siquiera con el mapa estatico."
                )
                return
        else:
            self.consecutive_static_fallbacks = 0

        self.publish_path(path_cells)

        self.last_plan_pose = (start_x, start_y)

        self.get_logger().info(f"Camino Theta* encontrado con {len(path_cells)} waypoints.")

    def heuristic(self, a, b):
        """
        Distancia euclidiana entre dos celdas.
        """
        ar, ac = a
        br, bc = b
        return math.sqrt((ar - br) ** 2 + (ac - bc) ** 2)

    def get_neighbors(self, cell):
        """
        Vecinos 8-conectados.
        Permite moverse horizontal, vertical y diagonal.

        Para movimientos diagonales, exige que las dos celdas ortogonales
        adyacentes tambien esten libres. Si no, el camino "corta la esquina"
        pasando entre dos obstaculos que se tocan en diagonal, algo que el
        robot (que tiene volumen) no puede hacer en la realidad.
        """
        row, col = cell

        directions = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2)),
        ]

        neighbors = []

        for dr, dc, cost in directions:
            nr = row + dr
            nc = col + dc
            new_cell = (nr, nc)

            if not self.is_free(new_cell):
                continue

            if dr != 0 and dc != 0:
                if not self.is_free((row + dr, col)) or not self.is_free((row, col + dc)):
                    continue

            neighbors.append((new_cell, cost))

        return neighbors
    
    #A* -> capaz mas adelante lo cambio por theta* para que sea mas sueave
    # def a_star(self, start, goal):
    #     """
    #     A* sobre la grilla inflada.
    #     """
    #     open_heap = []
    #     heapq.heappush(open_heap, (0.0, start))

    #     came_from = {}
    #     g_score = {start: 0.0}

    #     closed_set = set()

    #     while open_heap:
    #         _, current = heapq.heappop(open_heap)

    #         if current in closed_set:
    #             continue

    #         if current == goal:
    #             return self.reconstruct_path(came_from, current)

    #         closed_set.add(current)

    #         for neighbor, move_cost in self.get_neighbors(current):
    #             tentative_g = g_score[current] + move_cost

    #             if neighbor not in g_score or tentative_g < g_score[neighbor]:
    #                 came_from[neighbor] = current
    #                 g_score[neighbor] = tentative_g

    #                 f_score = tentative_g + self.heuristic(neighbor, goal)
    #                 heapq.heappush(open_heap, (f_score, neighbor))

    #     return None
    
    def theta_star(self, start, goal):
        """
        Theta* sobre la grilla inflada.

        Es parecido a A*, pero intenta conectar cada vecino directamente
        con el padre del nodo actual si existe linea de vision libre.

        Eso reduce waypoints innecesarios y genera caminos mas rectos.
        """
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))

        came_from = {}
        came_from[start] = start

        g_score = {start: 0.0}

        closed_set = set()

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            if current == goal:
                return self.reconstruct_theta_path(came_from, current)

            closed_set.add(current)

            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue

                parent = came_from[current]

                # Caso 1: si el padre de current ve directamente al vecino,
                # conectamos vecino con ese padre.
                if self.line_of_sight(parent, neighbor):
                    tentative_g = g_score[parent] + self.euclidean_cell_distance(parent, neighbor)

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = parent
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_heap, (f_score, neighbor))

                # Caso 2: si no hay linea de vision, hacemos como A* normal.
                else:
                    tentative_g = g_score[current] + move_cost

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_heap, (f_score, neighbor))

        return None


    def euclidean_cell_distance(self, a, b):
        """
        Distancia euclidiana entre dos celdas.
        """
        ar, ac = a
        br, bc = b
        return math.sqrt((ar - br) ** 2 + (ac - bc) ** 2)


    def reconstruct_theta_path(self, came_from, current):
        """
        Reconstruye el camino siguiendo padres.
        Igual que A*, pero lo dejamos separado para claridad.
        """
        path = [current]

        while came_from[current] != current:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path


    def line_of_sight(self, cell_a, cell_b):
        """
        Verifica si hay linea de vision libre entre dos celdas.

        Usamos una version simple de Bresenham:
        recorremos las celdas entre A y B y verificamos que todas sean libres
        en la grilla inflada.
        """
        r0, c0 = cell_a
        r1, c1 = cell_b

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)

        step_r = 1 if r1 > r0 else -1
        step_c = 1 if c1 > c0 else -1

        error = dr - dc

        r = r0
        c = c0

        while True:
            if not self.is_free((r, c)):
                return False

            if r == r1 and c == c1:
                break

            error2 = 2 * error
            moved_r = False
            moved_c = False

            if error2 > -dc:
                error -= dc
                r += step_r
                moved_r = True

            if error2 < dr:
                error += dr
                c += step_c
                moved_c = True

            # Si el paso de Bresenham avanzo en diagonal (r y c a la vez),
            # exigimos que las dos celdas ortogonales tambien esten libres
            # para no "cortar" la esquina entre dos obstaculos diagonales.
            if moved_r and moved_c:
                if not self.is_free((r - step_r, c)) or not self.is_free((r, c - step_c)):
                    return False

        return True

    def reconstruct_path(self, came_from, current):
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    # ============================================================
    # PUBLICACION
    # ============================================================

    def publish_path(self, path_cells):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        # Orientacion del goal final, para conservar el yaw deseado.
        final_orientation = self.goal_pose.pose.orientation

        for i, (row, col) in enumerate(path_cells):
            x, y = self.map_to_world(row, col)

            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0

            # Por ahora no calculamos orientacion intermedia.
            # Para los puntos intermedios dejamos identidad.
            pose.pose.orientation.w = 1.0

            # En el ultimo punto guardamos la orientacion final del goal.
            if i == len(path_cells) - 1:
                pose.pose.orientation = final_orientation

            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)

    node = PathPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()