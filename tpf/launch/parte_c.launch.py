import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    """
    Arma los remappings de topicos del robot (scan/odom/cmd_vel) en base al
    argumento 'robot_namespace', resuelto recien aca (en tiempo de launch)
    para poder construir los nombres de topico con un simple f-string en vez
    de encadenar substitutions. Asi el dia del laboratorio, si el robot no
    publica bajo /tb4_0/..., alcanza con
    'ros2 launch tpf parte_c.launch.py robot_namespace:=""' (o el namespace
    que corresponda) sin tocar el codigo.
    """
    pkg_tpf = get_package_share_directory("tpf")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map_yaml")
    exploration_waypoints = LaunchConfiguration("exploration_waypoints")

    robot_namespace = LaunchConfiguration("robot_namespace").perform(context).strip("/")
    prefix = f"/{robot_namespace}" if robot_namespace else ""

    scan_topic = f"{prefix}/scan"
    odom_topic = f"{prefix}/odom"
    cmd_vel_topic = f"{prefix}/cmd_vel"

    # TF temporal map -> odom (igual que parte_b.launch.py; particle_localizer
    # no publica esta TF, solo el topico /estimated_pose).
    static_tf_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_map_to_odom",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"yaml_filename": map_yaml},
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map_server",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    particle_localizer = Node(
        package="tpf",
        executable="particle_localizer",
        name="particle_localizer",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time},
                   {"mode": "real"},
                   ],
        remappings=[
            ("/odom", odom_topic),
            ("/scan", scan_topic),
        ],
    )

    path_planner = Node(
        package="tpf",
        executable="path_planner",
        name="path_planner",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time},
                    {"mode": "real"}
                    ],
        remappings=[
            ("/scan", scan_topic),
        ],
    )

    path_follower = Node(
        package="tpf",
        executable="path_follower",
        name="path_follower",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[
            ("/cmd_vel", cmd_vel_topic),
        ],
    )

    navigation_manager = Node(
        package="tpf",
        executable="navigation_manager",
        name="navigation_manager",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    # obstacle_avoidance.py arma sus propios nombres de topico internamente
    # a partir de robot_type ("tb4" -> /tb4_0/scan y /tb4_0/cmd_vel). Se
    # remapea desde esos nombres fijos hacia los topicos calculados arriba;
    # con robot_namespace="tb4_0" (default) el remap es un no-op.
    obstacle_avoidance = Node(
        package="tpf",
        executable="obstacle_avoidance",
        name="obstacle_avoidance",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"robot_type": "tb4"},
        ],
        remappings=[
            ("/tb4_0/scan", scan_topic),
            ("/tb4_0/cmd_vel", cmd_vel_topic),
        ],
    )

    aruco_detector = Node(
        package="tpf",
        executable="aruco_detector",
        name="aruco_detector",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"save_csv": False},
        ],
    )

    cone_detector = Node(
        package="tpf",
        executable="cone_detector",
        name="cone_detector",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    cone_mission_manager = Node(
        package="tpf",
        executable="cone_mission_manager",
        name="cone_mission_manager",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"exploration_waypoints": exploration_waypoints},
        ],
    )

    # rviz se retrasa 5s (igual que parte_b.launch.py): si arranca al mismo
    # tiempo que el resto de los nodos, su inicializacion de OpenGL compite
    # por CPU con la transicion de lifecycle Configure->Activate de
    # map_server y puede hacer que esa transicion nunca llegue a tiempo
    # (map_server queda trabado en "unconfigured" y /map nunca se publica).
    rviz_config = os.path.join(pkg_tpf, "rviz", "parte_c.rviz")
    rviz = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time}],
            )
        ],
    )

    return [
        static_tf_map_to_odom,
        map_server,
        lifecycle_manager,
        particle_localizer,
        path_planner,
        path_follower,
        navigation_manager,
        obstacle_avoidance,
        aruco_detector,
        cone_detector,
        cone_mission_manager,
        rviz,
    ]


def generate_launch_description():
    """
    Parte C, Fase 2: navegacion (Parte B, sin cambios de codigo) + mision
    de busqueda de conos (nodos nuevos de Parte C), contra el robot real
    TB4 (namespace configurable via robot_namespace) o contra un rosbag
    reproducido con --clock.

    No levanta Gazebo: los sensores vienen del bag (`ros2 bag play
    tpf/rosbags/laberinto_conos --clock`) o del stack real del TB4,
    asumido ya corriendo por fuera de este launch.
    """

    pkg_tpf = get_package_share_directory("tpf")

    default_map_yaml = os.path.join(pkg_tpf, "maps", "map.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="true para bag/replay, false para el robot real en vivo.",
    )
    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value=default_map_yaml,
        description="Mapa del laberinto real, generado en la Fase 1 (mapeo).",
    )
    exploration_waypoints_arg = DeclareLaunchArgument(
        "exploration_waypoints",
        default_value="",
        description=(
            "Waypoints de exploracion como \"x0,y0,yaw0;x1,y1,yaw1;...\" "
            "(definidos a mano sobre el map_yaml real una vez generado)."
        ),
    )
    robot_namespace_arg = DeclareLaunchArgument(
        "robot_namespace",
        default_value="tb4_0",
        description=(
            "Namespace/prefijo de los topicos del robot real (scan, odom, "
            "cmd_vel). Pasar \"\" (vacio) si el robot publica directamente "
            "en /scan, /odom y /cmd_vel sin namespace."
        ),
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_yaml_arg,
        exploration_waypoints_arg,
        robot_namespace_arg,
        OpaqueFunction(function=launch_setup),
    ])
