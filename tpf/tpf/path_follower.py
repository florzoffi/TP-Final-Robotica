import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from std_msgs.msg import String

def yaw_from_quaternion(q):
    """
    Convierte quaternion ROS a yaw.
    """
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """
    Lleva un angulo al rango [-pi, pi].
    """
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class PathFollower(Node):

    def __init__(self):
        super().__init__("path_follower")

        # Path 
        self.path = []

        self.current_waypoint_index = 0
        self.current_pose = None

        # Tolerancia para considerar que llegue a un waypoint
        self.position_tolerance = 0.10

        # Si el error angular es grande, primero giro antes de avanzar
        self.angle_tolerance = 0.20

        # Velocidades maximas
        self.max_linear_speed = 0.12
        self.max_angular_speed = 0.6

        # Ganancias simples tipo proporcional
        self.k_linear = 0.5
        self.k_angular = 1.5
        
        self.final_yaw = None
        self.final_angle_tolerance = 0.08
        self.goal_reached = False

        self.nav_state = ""

        self.pose_sub = self.create_subscription(
            PoseStamped,
            "/estimated_pose",
            self.pose_callback,
            10,
        )

        self.path_sub = self.create_subscription(
            Path,
            "/plan",
            self.path_callback,
            10,
        )

        self.nav_state_sub = self.create_subscription(
            String,
            "/navigation_state",
            self.nav_state_callback,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

        self.control_timer = self.create_timer(
            0.1,
            self.control_loop,
        )

        self.get_logger().info("Path follower iniciado con path hardcodeado.")

    def nav_state_callback(self, msg):
        self.nav_state = msg.data

    def pose_callback(self, msg):
        self.current_pose = msg
        
    
        
    def path_callback(self, msg):
        """
        Guarda el camino recibido desde el planner.
        Cuando llega un nuevo plan, no arrancamos necesariamente desde el waypoint 0,
        porque ese punto suele ser la celda actual del robot y puede quedar atras.
        """
        self.path = []

        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.path.append((x, y))

        if len(msg.poses) > 0:
            self.final_yaw = yaw_from_quaternion(msg.poses[-1].pose.orientation)

        self.goal_reached = False

        if len(self.path) == 0:
            self.current_waypoint_index = 0
            return

        # Si todavía no tengo pose, arranco salteando el primer punto si puedo.
        if self.current_pose is None:
            self.current_waypoint_index = 1 if len(self.path) > 1 else 0
        else:
            robot_x = self.current_pose.pose.position.x
            robot_y = self.current_pose.pose.position.y

            # Busco el waypoint más cercano al robot.
            closest_index = 0
            closest_dist = float("inf")

            for i, (x, y) in enumerate(self.path):
                dx = x - robot_x
                dy = y - robot_y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < closest_dist:
                    closest_dist = dist
                    closest_index = i

            # Arranco en el siguiente waypoint, para no intentar volver al punto inicial.
            self.current_waypoint_index = min(closest_index + 1, len(self.path) - 1)

        self.get_logger().info(
            f"Nuevo path recibido con {len(self.path)} waypoints. "
            f"Arrancando desde waypoint {self.current_waypoint_index}."
        )

    def control_loop(self):
        """
        Loop de control que corre cada 0.1 s.
        Decide que velocidad publicar en /cmd_vel.
        """
        if self.nav_state == "AVOIDING_OBSTACLE":
            # Obstacle avoidance node owns cmd_vel right now.
            return

        if self.current_pose is None:
            return

        if len(self.path) == 0:
            return

        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        robot_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)

        if self.current_waypoint_index >= len(self.path):
            self.align_to_final_yaw(robot_yaw)
            return
        

        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        robot_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)

        target_x, target_y = self.path[self.current_waypoint_index]

        dx = target_x - robot_x
        dy = target_y - robot_y

        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)
        angle_error = normalize_angle(target_angle - robot_yaw)

        # Si llegue al waypoint, paso al siguiente.
        if distance < self.position_tolerance:
            self.get_logger().info(
                f"Waypoint {self.current_waypoint_index} alcanzado."
            )
            self.current_waypoint_index += 1

            if self.current_waypoint_index >= len(self.path):
                self.get_logger().info("Path completo alcanzado. Frenando.")
                self.stop_robot()
            

            return

        cmd = Twist()

        # Si estoy muy mal orientado, giro en el lugar.
        if abs(angle_error) > self.angle_tolerance:
            cmd.linear.x = 0.0
            cmd.angular.z = self.k_angular * angle_error
        else:
            # Si estoy razonablemente orientado, avanzo y corrijo giro.
            cmd.linear.x = self.k_linear * distance
            cmd.angular.z = self.k_angular * angle_error

        # Saturaciones para no mandar velocidades demasiado grandes.
        cmd.linear.x = max(
            -self.max_linear_speed,
            min(self.max_linear_speed, cmd.linear.x),
        )

        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, cmd.angular.z),
        )

        self.cmd_pub.publish(cmd)

    def align_to_final_yaw(self, robot_yaw):
        """
        Una vez que el robot llego al ultimo waypoint,
        gira en el lugar hasta alcanzar la orientacion final del goal.
        """
        if self.final_yaw is None:
            self.stop_robot()
            return

        angle_error = normalize_angle(self.final_yaw - robot_yaw)

        if abs(angle_error) < self.final_angle_tolerance:
            if not self.goal_reached:
                self.get_logger().info("Objetivo alcanzado con orientacion final correcta.")
                self.goal_reached = True

            self.stop_robot()
            return

        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = self.k_angular * angle_error

        cmd.angular.z = max(
            -self.max_angular_speed,
            min(self.max_angular_speed, cmd.angular.z),
        )

        self.cmd_pub.publish(cmd)
    
    
    def stop_robot(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)

    node = PathFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()