import math
import heapq

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped



class PathPlanner(Node):

    def __init__(self):
        super().__init__("path_planner")

        # Radio de seguridad alrededor de paredes onda para no acercarnos
        # El robot no es un punto, entonces inflamos obstáculos.
        self.inflation_radius_m = 0.22

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
        self.replan_distance_threshold = 0.25

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

        self.map_received = True

        self.get_logger().info(
            f"Mapa recibido para planificacion: {self.map_width}x{self.map_height}, "
            f"res={self.map_resolution:.3f}"
        )

    def pose_callback(self, msg):
        self.current_pose = msg

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
            self.get_logger().warn("La celda inicial esta ocupada o demasiado cerca de una pared.")
            return

        if not self.is_free(goal_cell):
            self.get_logger().warn("La celda objetivo esta ocupada o demasiado cerca de una pared.")
            return

        path_cells = self.theta_star(start_cell, goal_cell)

        if path_cells is None:
            self.get_logger().warn("No se encontro camino.")
            return

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

            if self.is_free(new_cell):
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

            if error2 > -dc:
                error -= dc
                r += step_r

            if error2 < dr:
                error += dr
                c += step_c

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