import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from rclpy.qos import QoSProfile, ReliabilityPolicy
import math


class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__( 'obstacle_avoidance' )
        self.declare_parameter( 'robot_type', 'tb4' )
        self.robot_type = self.get_parameter( 'robot_type' ).value
        qos_tb4 = QoSProfile( depth=10, reliability=ReliabilityPolicy.BEST_EFFORT )

        if self.robot_type == 'tb4':
            namespace = '/tb4_0'
            self.cmd_topic = namespace + '/cmd_vel'
            self.scan_topic = namespace + '/scan'
            self.odom_topic = namespace + '/odom'
            self.sub_qos = qos_tb4
            self.front_index_offset = 90
            self.use_intensity_filter = True
            self.turn_angle = math.radians( 110 - 20.5 )
            self.linear_speed = 0.5
            self.angular_speed = 1
        else:
            namespace = ''
            self.cmd_topic = '/cmd_vel'
            self.scan_topic = '/scan'
            self.odom_topic = '/calc_odom'
            self.sub_qos = 10
            self.front_index_offset = 0
            self.use_intensity_filter = False
            self.turn_angle = math.radians( 110 )
            self.linear_speed = 0.5
            self.angular_speed = 1.0

        self.min_distance = 0.5
        self.window = 20
        self.obstacle_detected = False
        self.rotating = False
        self.current_yaw = 0.0
        self.start_yaw = None
        self.cmd_pub = self.create_publisher( Twist, self.cmd_topic, 10 )
        self.scan_sub = self.create_subscription( LaserScan, self.scan_topic, self.scan_callback, self.sub_qos )
        self.odom_sub = self.create_subscription( Odometry, self.odom_topic, self.odom_callback, self.sub_qos )
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info( f'Running obstacle avoidance for {self.robot_type}' )
        self.get_logger().info( f'cmd_vel topic: {self.cmd_topic}' )
        self.get_logger().info( f'scan topic: {self.scan_topic}' )
        self.get_logger().info( f'odom topic: {self.odom_topic}' )

    def scan_callback( self, msg ):
        n = len( msg.ranges )
        if self.robot_type == 'tb4':
            front_index = n // 4
        else:
            front_index = 0

        valid_ranges = []
        for i in range( front_index - self.window, front_index + self.window + 1 ):
            index = i % n
            r = msg.ranges[index]
            if self.use_intensity_filter and len( msg.intensities ) > index:
                if msg.intensities[index] == 0.0:
                    continue
            if not math.isinf( r ) and not math.isnan( r ):
                valid_ranges.append( r )

        if len( valid_ranges ) == 0:
            self.obstacle_detected = False
            return
        self.obstacle_detected = min( valid_ranges ) <= self.min_distance

    def odom_callback( self, msg ):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * ( q.w * q.z + q.x * q.y )
        cosy_cosp = 1 - 2 * ( q.y * q.y + q.z * q.z )
        self.current_yaw = math.atan2( siny_cosp, cosy_cosp )

    def angle_diff( self, current, start ):
        diff = current - start

        while diff > math.pi:
            diff -= 2 * math.pi

        while diff < -math.pi:
            diff += 2 * math.pi

        return abs( diff )

    def control_loop( self ):
        cmd = Twist()

        if self.rotating:
            rotated_angle = self.angle_diff( self.current_yaw, self.start_yaw )
            if rotated_angle < self.turn_angle:
                cmd.angular.z = self.angular_speed
            else:
                self.rotating = False
                self.start_yaw = None
                cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            return
        
        if self.obstacle_detected:
            self.rotating = True
            self.start_yaw = self.current_yaw
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            cmd.linear.x = self.linear_speed
        self.cmd_pub.publish( cmd )


def main( args=None ):
    rclpy.init( args=args )
    node = ObstacleAvoidance()
    rclpy.spin( node )
    node.destroy_node()
    rclpy.shutdown()