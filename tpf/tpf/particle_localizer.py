import math
import random
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan


def yaw_from_quaternion(q):
    """
    Convierte un quaternion ROS a yaw.
    Como el robot se mueve en 2D, solo nos interesa el angulo alrededor de Z.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw):
    """
    Convierte yaw a quaternion ROS.
    Roll y pitch son cero porque trabajamos en 2D.
    """
    q = Pose().orientation
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def normalize_angle(angle):
    """
    Lleva un angulo al rango [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ParticleLocalizer(Node):

    def __init__(self):
        super().__init__("particle_localizer")

        self.num_particles = 500

        # Parametros de ruido
        self.init_std_xy = 0.30
        self.init_std_yaw = 0.8
        self.motion_std_distance = 0.01
        self.motion_std_yaw = 0.01

        # Parametros del modelo de observacion con LIDAR
        self.laser_step = 10 #uso 1 d cada 10 rayos del lidar
        self.sensor_sigma = 0.20 #tolerancia contra paraedes, mas chico mas estricto
        
        self.max_likelihood_dist = 1.0 # Distancia maxima que nos importa hasta la pared mas cercana
        
        self.initialized = False
        self.particles = []     #cada particula la guardamos como [x, y, yaw] en coordenadas del mapa
        self.weights = []       #peso de cada particula segun que tan bien coincide con el lidar
        
        #esta es el x, y y yaw de la odometria anterior, para poder calcular cuanto se movio el robot desde la ultima odometria
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None

        self.latest_scan = None

        # Datos del mapa. Los llenamos cuando llegue /map.
        self.map_received = False
        self.map_width = None
        self.map_height = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.occupancy_data = None
        self.distance_field = None

        # /map viene de nav2_map_server y usa QoS transient local.
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

        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose", #Cuando en rviz tocamos el botoncito de 2D Pose Estimate se publica en este topico que nos dice donde estamos parados
            self.initialpose_callback,
            10,
        )

        # El TB4 real (y los rosbags de Parte C) publican /tb4_0/odom con QoS
        # BEST_EFFORT — un subscriber RELIABLE (el default al pasar un int)
        # queda incompatible y nunca recibe nada. BEST_EFFORT acá es seguro
        # también contra Gazebo (su /odom es RELIABLE, y BEST_EFFORT matchea
        # con cualquier publisher).
        odom_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom", #mi nueva odometria que me llega del robot
            self.odom_callback,
            odom_qos,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan", #scan del lidar, lo usamos para corregir las particulas contra el mapa
            self.scan_callback,
            10,
        )

        self.estimated_pose_pub = self.create_publisher(
            PoseStamped,
            "/estimated_pose", #la pose estimada es el promedio de todas las particulas
            10,
        )

        self.particle_cloud_pub = self.create_publisher(
            PoseArray,
            "/particle_cloud", #publica todas las particulas para verlas en rviz
            10,
        )

        self.get_logger().info("Particle localizer iniciado. Esperando /map e /initialpose...")

    def map_callback(self, msg):
        """
        Recibe el mapa /map.
        A partir del mapa construimos un distance field:
        para cada celda, guardamos aproximadamente qué tan lejos está de una pared.
        Eso después sirve para saber si un rayo del lidar cae cerca de una pared o no.
        """
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_resolution = msg.info.resolution
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y
        self.occupancy_data = list(msg.data)

        self.build_distance_field()

        self.map_received = True

        self.get_logger().info(
            f"Mapa recibido: {self.map_width}x{self.map_height}, "
            f"res={self.map_resolution:.3f} m/celda"
        )

    def build_distance_field(self):
        """
        Construye un campo de distancias usando BFS.

        Idea:
        - las paredes tienen distancia 0;
        - las celdas vecinas a paredes tienen distancia chica;
        - las celdas lejos de paredes tienen distancia grande.

        Esto evita hacer ray casting completo para cada particula.
        """
        width = self.map_width
        height = self.map_height
        res = self.map_resolution

        max_cells = int(self.max_likelihood_dist / res)

        dist_cells = [[None for _ in range(width)] for _ in range(height)]
        q = deque()

        for row in range(height):
            for col in range(width):
                idx = row * width + col
                occ = self.occupancy_data[idx]

                # En OccupancyGrid:
                # 0 = libre, 100 = ocupado, -1 = desconocido.
                # Para navegar seguro tratamos desconocido como obstaculo.
                if occ > 50 or occ == -1:
                    dist_cells[row][col] = 0
                    q.append((row, col))

        neighbors = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1),
        ]

        while q:
            row, col = q.popleft()

            if dist_cells[row][col] >= max_cells:
                continue

            for dr, dc in neighbors:
                nr = row + dr
                nc = col + dc

                if nr < 0 or nr >= height or nc < 0 or nc >= width:
                    continue

                if dist_cells[nr][nc] is None:
                    dist_cells[nr][nc] = dist_cells[row][col] + 1
                    q.append((nr, nc))

        self.distance_field = [
            [self.max_likelihood_dist for _ in range(width)]
            for _ in range(height)
        ]

        for row in range(height):
            for col in range(width):
                if dist_cells[row][col] is not None:
                    self.distance_field[row][col] = min(
                        dist_cells[row][col] * res,
                        self.max_likelihood_dist,
                    )

    def world_to_map(self, x, y):
        """
        Convierte coordenadas del mundo/mapa en metros a celda de grilla.
        Devuelve (row, col).
        """
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)

        if row < 0 or row >= self.map_height or col < 0 or col >= self.map_width:
            return None

        return row, col

    def distance_to_nearest_obstacle(self, x, y):
        """
        Dado un punto x,y del mapa, devuelve que tan lejos esta de la pared mas cercana.
        Si cae afuera del mapa, devolvemos distancia maxima.
        """
        cell = self.world_to_map(x, y)

        if cell is None:
            return self.max_likelihood_dist

        row, col = cell
        return self.distance_field[row][col]

    def initialpose_callback(self, msg):
        """
        Se ejecuta cuando el usuario usa 2D Pose Estimate en RViz.
        Crea una nube de particulas alrededor de esa pose inicial.
        """
        if not self.map_received:
            self.get_logger().warn("Llego /initialpose pero todavia no hay /map. Ignorando.")
            return

        x0 = msg.pose.pose.position.x
        y0 = msg.pose.pose.position.y
        yaw0 = yaw_from_quaternion(msg.pose.pose.orientation)

        self.particles = []
        self.weights = []

        for _ in range(self.num_particles):
            x = random.gauss(x0, self.init_std_xy)
            y = random.gauss(y0, self.init_std_xy)
            yaw = random.gauss(yaw0, self.init_std_yaw)
            self.particles.append([x, y, normalize_angle(yaw)])
            self.weights.append(1.0 / self.num_particles)

        self.initialized = True
        self.last_odom_x = None
        self.last_odom_y = None
        self.last_odom_yaw = None
        self.publish_outputs(msg.header.stamp)

        self.get_logger().info(
            f"Inicializado con {self.num_particles} particulas alrededor de "
            f"x={x0:.2f}, y={y0:.2f}, yaw={yaw0:.2f}"
        )

    def odom_callback(self, msg):
        """
        Se ejecuta cada vez que llega odometria.
        Calcula cuanto se movio el robot desde la ultima odometria
        y aplica ese movimiento a todas las particulas.
        """
        if not self.initialized:
            return

        #odometria actual
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.last_odom_x is None: #si esta es la primera no calculamos una nueva, esta es la nueva
            self.last_odom_x = odom_x
            self.last_odom_y = odom_y
            self.last_odom_yaw = odom_yaw
            return

        #aca sivemos cuanto cambia la odometria
        dx = odom_x - self.last_odom_x
        dy = odom_y - self.last_odom_y
        dyaw = normalize_angle(odom_yaw - self.last_odom_yaw)

        #aca si, calculamos la distancia y orientacion en base al cambio
        distance = math.sqrt(dx * dx + dy * dy) #distanca euclidia
        movement_angle = math.atan2(dy, dx) if distance > 1e-6 else self.last_odom_yaw #atan2 que era para ver la diferencia de anuglos

        # Este angulo dice hacia donde fue el movimiento respecto de la orientacion vieja del robot.
        relative_angle = normalize_angle(movement_angle - self.last_odom_yaw)

        #aplicamos el mismo movimiento a las particulas
        for p in self.particles:
            x, y, yaw = p

            #cuantos angulos se movio y actualizacion
            particle_move_angle = yaw + relative_angle

            # Agregamos un poco de ruido para representar incertidumbre
            noisy_distance = distance + random.gauss(0.0, self.motion_std_distance)
            noisy_dyaw = dyaw + random.gauss(0.0, self.motion_std_yaw)
            
            #efectivamente aplicamos el cambio a la particula
            x += noisy_distance * math.cos(particle_move_angle)
            y += noisy_distance * math.sin(particle_move_angle)
            yaw = normalize_angle(yaw + noisy_dyaw)

            p[0] = x
            p[1] = y
            p[2] = yaw

        #actualizamos la odometria vieja por esta
        self.last_odom_x = odom_x
        self.last_odom_y = odom_y
        self.last_odom_yaw = odom_yaw

        self.publish_outputs(msg.header.stamp)


    def scan_callback(self, msg):
        """
        Se ejecuta cada vez que llega /scan.
        Esta es la parte nueva: usa LIDAR + mapa para decidir qué partículas son mejores.
        """
        self.latest_scan = msg

        if not self.initialized:
            return

        if not self.map_received:
            return

        self.update_weights_with_scan(msg)

        # N_eff se calcula ANTES de resamplear, cuando los pesos todavia son
        # no-uniformes. Calcularlo despues es un bug: resample_particles()
        # resetea self.weights a uniforme, asi que daria siempre num_particles.
        sum_sq = sum(w * w for w in self.weights)
        n_eff = 1.0 / sum_sq if sum_sq > 0 else float(self.num_particles)

        self.resample_particles()

        xs = [p[0] for p in self.particles]
        ys = [p[1] for p in self.particles]
        yaws = [p[2] for p in self.particles]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        std_x = math.sqrt(sum((v - mean_x) ** 2 for v in xs) / len(xs))
        std_y = math.sqrt(sum((v - mean_y) ** 2 for v in ys) / len(ys))
        std_yaw = math.sqrt(sum((v - mean_yaw) ** 2
                               for v in yaws
                               for mean_yaw in [sum(yaws) / len(yaws)]) / len(yaws))

        self.get_logger().info(
            f"DIAG localizer: N_eff={n_eff:.0f}/{self.num_particles} | "
            f"pose_est=({mean_x:.2f},{mean_y:.2f}) | "
            f"std=({std_x:.2f}m, {std_y:.2f}m, {math.degrees(std_yaw):.1f}deg)",
            throttle_duration_sec=2.0,
        )

        self.publish_outputs(msg.header.stamp)

    def update_weights_with_scan(self, scan):
        """
        Calcula los pesos de las particulas.

        Para cada particula:
        - imagino que el robot esta en x,y,yaw de esa particula;
        - agarro algunos rayos reales del lidar;
        - proyecto donde terminaria cada rayo en el mapa;
        - si ese punto cae cerca de una pared del mapa, esa particula es mas creible.
        """
        log_weights = []

        # DIAG: para la primera particula, loguear detalle rayo a rayo
        diag_first = True

        for p_idx, (x, y, yaw) in enumerate(self.particles):
            log_w = 0.0
            valid_beams = 0
            beams_out_of_map = 0
            sample_dists = []   # para el diag de la primera particula

            for i in range(0, len(scan.ranges), self.laser_step):
                r = scan.ranges[i]

                if math.isinf(r) or math.isnan(r):
                    continue

                if r < scan.range_min or r > scan.range_max:
                    continue

                angle = scan.angle_min + i * scan.angle_increment
                # El mapa fue construido por corrected_map_node.py aplicando
                # LIDAR_ANGLE_OFFSET = pi/2. Sin este mismo offset aqui, los
                # rayos se proyectan 90° en la direccion equivocada respecto al
                # mapa → el filtro converge al yaw incorrecto.
                global_angle = yaw + math.pi / 2 + angle

                hit_x = x + r * math.cos(global_angle)
                hit_y = y + r * math.sin(global_angle)

                dist = self.distance_to_nearest_obstacle(hit_x, hit_y)

                if dist >= self.max_likelihood_dist:
                    beams_out_of_map += 1

                log_w += -0.5 * (dist / self.sensor_sigma) ** 2
                valid_beams += 1

                if diag_first and len(sample_dists) < 5:
                    sample_dists.append((round(hit_x, 2), round(hit_y, 2), round(dist, 3)))

            if diag_first:
                self.get_logger().info(
                    f"DIAG scan p0: particula=({x:.2f},{y:.2f},{math.degrees(yaw):.1f}deg) | "
                    f"rayos_validos={valid_beams} de {len(scan.ranges)//self.laser_step} | "
                    f"rayos_fuera_mapa={beams_out_of_map} | "
                    f"sample_hits(hit_x,hit_y,dist_pared)={sample_dists}",
                    throttle_duration_sec=2.0,
                )
                diag_first = False

            if valid_beams == 0:
                log_w = -100.0
            else:
                log_w = log_w / valid_beams

            log_weights.append(log_w)

        # Normalizacion estable
        max_log_w = max(log_weights)
        min_log_w = min(log_weights)

        # DIAG: si max-min es muy chico, todos los pesos van a quedar uniformes
        # y N_eff se mantendra en num_particles. La causa tipica: todos los rayos
        # caen fuera del mapa (dist=max_likelihood_dist en todos) o el mapa no
        # corresponde al bag y el LIDAR nunca "ve" paredes que coincidan.
        self.get_logger().info(
            f"DIAG pesos log: max={max_log_w:.4f} min={min_log_w:.4f} "
            f"spread={max_log_w - min_log_w:.4f} "
            f"(si spread < 0.01 -> pesos uniformes -> N_eff=N, filtro no discrimina)",
            throttle_duration_sec=2.0,
        )

        weights = [math.exp(lw - max_log_w) for lw in log_weights]
        total = sum(weights)

        if total <= 0.0:
            self.weights = [1.0 / self.num_particles for _ in range(self.num_particles)]
        else:
            self.weights = [w / total for w in weights]

    def resample_particles(self):
        """
        Resampling sistematico.
        Las particulas con peso alto se copian mas veces.
        Las particulas con peso bajo desaparecen.
        """
        new_particles = []

        step = 1.0 / self.num_particles
        start = random.uniform(0.0, step)
        positions = [start + i * step for i in range(self.num_particles)]

        cumulative = []
        total = 0.0

        for w in self.weights:
            total += w
            cumulative.append(total)

        i = 0

        for pos in positions:
            while i < self.num_particles - 1 and pos > cumulative[i]:
                i += 1

            x, y, yaw = self.particles[i]

            # Copiamos con un poquito de ruido para que no queden todas identicas.
            new_particles.append([
                x + random.gauss(0.0, 0.01),
                y + random.gauss(0.0, 0.01),
                normalize_angle(yaw + random.gauss(0.0, 0.005)),
            ])

        self.particles = new_particles
        self.weights = [1.0 / self.num_particles for _ in range(self.num_particles)]


    #promedio de las poses de las particulas para estimar pose del robotico
    def estimate_pose(self):
        """
        Calcula la pose estimada como promedio de las particulas.
        Para yaw no se puede promediar directamente el angulo,
        por eso se promedian seno y coseno.
        """
        if not self.particles:
            return None

        x_mean = sum(p[0] for p in self.particles) / len(self.particles)
        y_mean = sum(p[1] for p in self.particles) / len(self.particles)

        sin_sum = sum(math.sin(p[2]) for p in self.particles)
        cos_sum = sum(math.cos(p[2]) for p in self.particles)
        yaw_mean = math.atan2(sin_sum, cos_sum)

        return x_mean, y_mean, yaw_mean

    def publish_outputs(self, stamp):
        """
        Publica:
        - /estimated_pose: pose promedio estimada
        - /particle_cloud: todas las particulas para verlas en RViz
        """
        estimate = self.estimate_pose() #aca calculamos pose estimada

        if estimate is None:
            return

        x, y, yaw = estimate

        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = x
        pose_msg.pose.position.y = y
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation = quaternion_from_yaw(yaw)

        self.estimated_pose_pub.publish(pose_msg)

        cloud_msg = PoseArray()
        cloud_msg.header.stamp = stamp
        cloud_msg.header.frame_id = "map"

        #agregamos cada particula que creamos
        for px, py, pyaw in self.particles:
            pose = Pose()
            pose.position.x = px
            pose.position.y = py
            pose.position.z = 0.0
            pose.orientation = quaternion_from_yaw(pyaw)
            cloud_msg.poses.append(pose)

        self.particle_cloud_pub.publish(cloud_msg) #omaigods este es el que las publica todasss


def main(args=None): 
    rclpy.init(args=args)

    node = ParticleLocalizer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()