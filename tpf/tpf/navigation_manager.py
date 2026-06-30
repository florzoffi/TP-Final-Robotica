import math
from enum import Enum

import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Bool
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class NavState(Enum):
    WAITING_INITIAL_POSE = "WAITING_INITIAL_POSE"
    LOCALIZED = "LOCALIZED"
    WAITING_GOAL = "WAITING_GOAL"
    PLANNING = "PLANNING"
    FOLLOWING_PATH = "FOLLOWING_PATH"
    AVOIDING_OBSTACLE = "AVOIDING_OBSTACLE"
    FINAL_ALIGNMENT = "FINAL_ALIGNMENT"
    GOAL_REACHED = "GOAL_REACHED"


class NavigationManager(Node):

    def __init__(self):
        super().__init__("navigation_manager")

        self.state = NavState.WAITING_INITIAL_POSE

        self.has_initial_pose = False
        self.has_estimated_pose = False
        self.has_goal = False
        self.has_plan = False

        self.current_pose = None
        self.goal_pose = None
        self.current_path = []

        self.position_tolerance = 0.12
        self.final_angle_tolerance = 0.10

        self.obstacle_active = False

        self.initialpose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            "/initialpose",
            self.initialpose_callback,
            10,
        )

        self.estimated_pose_sub = self.create_subscription(
            PoseStamped,
            "/estimated_pose",
            self.estimated_pose_callback,
            10,
        )

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_callback,
            10,
        )

        self.plan_sub = self.create_subscription(
            Path,
            "/plan",
            self.plan_callback,
            10,
        )

        self.obstacle_sub = self.create_subscription(
            Bool,
            "/obstacle_detected",
            self.obstacle_callback,
            10,
        )

        self.state_pub = self.create_publisher(
            String,
            "/navigation_state",
            10,
        )

        self.goal_pub = self.create_publisher(
            PoseStamped,
            "/goal_pose",
            10,
        )

        self.timer = self.create_timer(
            0.2,
            self.update_state,
        )

        self.get_logger().info("Navigation manager iniciado.")

    def initialpose_callback(self, msg):
        self.has_initial_pose = True
        self.get_logger().info("FSM: initialpose recibido.")

        if self.state == NavState.WAITING_INITIAL_POSE:
            self.set_state(NavState.LOCALIZED)

    def estimated_pose_callback(self, msg):
        self.current_pose = msg
        self.has_estimated_pose = True

    def goal_callback(self, msg):
        self.goal_pose = msg
        self.has_goal = True
        self.has_plan = False
        self.current_path = []

        self.get_logger().info("FSM: goal recibido.")

        # Don't interrupt active obstacle avoidance; replan when it clears.
        if self.state == NavState.AVOIDING_OBSTACLE:
            return

        if self.has_estimated_pose:
            self.set_state(NavState.PLANNING)
        else:
            self.set_state(NavState.WAITING_INITIAL_POSE)

    def obstacle_callback(self, msg):
        self.obstacle_active = msg.data

        if msg.data and self.state == NavState.FOLLOWING_PATH:
            self.set_state(NavState.AVOIDING_OBSTACLE)
            self.has_plan = False
            self.current_path = []
            # Replanificar AHORA: el LIDAR todavía ve el obstáculo, así que
            # el planner lo incorporará a la grilla dinámica y generará un desvío.
            if self.goal_pose is not None:
                self.get_logger().info("FSM: obstáculo detectado — replaneando con LIDAR actual.")
                self.goal_pub.publish(self.goal_pose)
            self.publish_state()

        elif not msg.data and self.state == NavState.AVOIDING_OBSTACLE:
            # El período de bloqueo terminó. Si el planner ya generó un plan
            # alternativo, lo seguimos directamente sin volver a PLANNING.
            if self.has_plan:
                self.get_logger().info("FSM: bloqueo terminado — siguiendo plan alternativo.")
                self.set_state(NavState.FOLLOWING_PATH)
            else:
                self.get_logger().info("FSM: bloqueo terminado — sin plan, replaneando.")
                self.set_state(NavState.PLANNING)
                if self.goal_pose is not None:
                    self.goal_pub.publish(self.goal_pose)
            self.publish_state()

    def plan_callback(self, msg):
        self.current_path = msg.poses
        self.has_plan = len(self.current_path) > 0

        if self.has_plan:
            self.get_logger().info(
                f"FSM: plan recibido con {len(self.current_path)} poses."
            )

            if self.state in [NavState.PLANNING, NavState.WAITING_GOAL, NavState.LOCALIZED]:
                self.set_state(NavState.FOLLOWING_PATH)

    def update_state(self):
        """
        Revisa periodicamente las condiciones del sistema y actualiza el estado.
        """
        if not self.has_initial_pose:
            self.set_state(NavState.WAITING_INITIAL_POSE)
            self.publish_state()
            return

        if not self.has_estimated_pose:
            self.set_state(NavState.LOCALIZED)
            self.publish_state()
            return

        if not self.has_goal:
            self.set_state(NavState.WAITING_GOAL)
            self.publish_state()
            return

        # Don't override obstacle avoidance via the periodic timer.
        if self.state == NavState.AVOIDING_OBSTACLE:
            self.publish_state()
            return

        if self.has_goal and not self.has_plan:
            self.set_state(NavState.PLANNING)
            self.publish_state()
            return

        if self.current_pose is None or self.goal_pose is None:
            self.publish_state()
            return

        distance_to_goal = self.compute_distance_to_goal()

        if distance_to_goal > self.position_tolerance:
            self.set_state(NavState.FOLLOWING_PATH)
            self.publish_state()
            return

        yaw_error = self.compute_final_yaw_error()

        if abs(yaw_error) > self.final_angle_tolerance:
            self.set_state(NavState.FINAL_ALIGNMENT)
            self.publish_state()
            return

        self.set_state(NavState.GOAL_REACHED)
        self.publish_state()

    def compute_distance_to_goal(self):
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y

        goal_x = self.goal_pose.pose.position.x
        goal_y = self.goal_pose.pose.position.y

        dx = goal_x - robot_x
        dy = goal_y - robot_y

        return math.sqrt(dx * dx + dy * dy)

    def compute_final_yaw_error(self):
        robot_yaw = yaw_from_quaternion(self.current_pose.pose.orientation)
        goal_yaw = yaw_from_quaternion(self.goal_pose.pose.orientation)
        return normalize_angle(goal_yaw - robot_yaw)

    def set_state(self, new_state):
        if new_state == self.state:
            return

        old_state = self.state
        self.state = new_state

        self.get_logger().info(
            f"FSM: {old_state.value} -> {new_state.value}"
        )

    def publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = NavigationManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()