import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


# ---------------------------------------------------------------------------
# Variante de parte_c.launch.py pensada para la computadora del laboratorio
# (Ubuntu con GPU/driver que hace crashear a rviz con "libGL error: failed to
# create drawable" -> segfault). Diferencias respecto de parte_c.launch.py:
#
#   1. rviz se lanza forzando renderizado por software (LIBGL_ALWAYS_SOFTWARE)
#      y X11 (QT_QPA_PLATFORM=xcb). Eso evita el crash de OpenGL en maquinas
#      sin GPU utilizable, VMs o sesiones Wayland.
#   2. Argumento 'use_rviz' (default true): permite correr TODO el pipeline
#      SIN rviz (use_rviz:=false), util si el render sigue fallando —  la
#      mision de conos no necesita rviz para funcionar, solo para visualizar.
#      Ojo: sin rviz no hay boton "2D Pose Estimate"; ver nota al pie.
#   3. El mapa por defecto sale del paquete instalado (maps2/map2.yaml), asi
#      no hay que pasar rutas absolutas tipo $HOME/ws/... que cambian de
#      maquina en maquina (fue justo lo que fallo en la compu del lab).
#   4. robot_namespace por defecto = "tb4_1" (el robot del laboratorio).
# ---------------------------------------------------------------------------


def launch_setup(context, *args, **kwargs):
    pkg_tpf = get_package_share_directory("tpf")

    use_sim_time = LaunchConfiguration("use_sim_time")
    map_yaml = LaunchConfiguration("map_yaml")
    exploration_waypoints = LaunchConfiguration("exploration_waypoints")

    robot_namespace = LaunchConfiguration("robot_namespace").perform(context).strip("/")
    prefix = f"/{robot_namespace}" if robot_namespace else ""

    use_rviz = LaunchConfiguration("use_rviz").perform(context).lower() in ("true", "1", "yes")

    scan_topic = f"{prefix}/scan"
    odom_topic = f"{prefix}/odom"
    cmd_vel_topic = f"{prefix}/cmd_vel"
    camera_image_topic = f"{prefix}/oakd/rgb/preview/image_raw"
    camera_info_topic = f"{prefix}/oakd/rgb/preview/camera_info"

    # TF temporal map -> odom (particle_localizer no publica esta TF, solo el
    # topico /estimated_pose).
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
        remappings=[
            ("/tb4_0/oakd/rgb/preview/image_raw", camera_image_topic),
        ],
    )

    cone_detector = Node(
        package="tpf",
        executable="cone_detector",
        name="cone_detector",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=[
            ("/tb4_0/oakd/rgb/preview/image_raw", camera_image_topic),
            ("/tb4_0/oakd/rgb/preview/camera_info", camera_info_topic),
        ],
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

    actions = [
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
    ]

    if use_rviz:
        # additional_env fuerza render por software: LIBGL_ALWAYS_SOFTWARE=1
        # hace que Mesa use llvmpipe (CPU) en vez de la GPU, evitando el
        # "failed to create drawable" / segfault. QT_QPA_PLATFORM=xcb fuerza
        # X11 (la maquina del lab avisaba XDG_SESSION_TYPE=wayland). Es mas
        # lento que la GPU pero estable; por eso el rviz de esta variante usa
        # una config liviana (parte_c_lab.rviz) con pocos displays.
        rviz_config = os.path.join(pkg_tpf, "rviz", "parte_c_lab.rviz")
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
                    additional_env={
                        "LIBGL_ALWAYS_SOFTWARE": "1",
                        "QT_QPA_PLATFORM": "xcb",
                        "MESA_GL_VERSION_OVERRIDE": "3.3",
                    },
                )
            ],
        )
        actions.append(rviz)

    return actions


def generate_launch_description():
    pkg_tpf = get_package_share_directory("tpf")

    # Default desde el paquete instalado, no una ruta absoluta ~/ws/... que
    # cambia de maquina. En cualquier compu donde este paquete este instalado
    # con maps2, esto resuelve solo.
    default_map_yaml = os.path.join(pkg_tpf, "maps2", "map2.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="true para bag/replay, false para el robot real en vivo.",
    )
    map_yaml_arg = DeclareLaunchArgument(
        "map_yaml",
        default_value=default_map_yaml,
        description="Mapa del laberinto. Default: maps2/map2.yaml del paquete.",
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
        default_value="tb4_1",
        description=(
            "Namespace/prefijo de los topicos del robot (scan, odom, cmd_vel, "
            "camara). Default tb4_1 (robot del laboratorio). Pasar \"\" si "
            "publica sin namespace."
        ),
    )
    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description=(
            "true lanza rviz con render por software. Pasar use_rviz:=false "
            "para correr el pipeline sin rviz si el render igual falla."
        ),
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_yaml_arg,
        exploration_waypoints_arg,
        robot_namespace_arg,
        use_rviz_arg,
        OpaqueFunction(function=launch_setup),
    ])
