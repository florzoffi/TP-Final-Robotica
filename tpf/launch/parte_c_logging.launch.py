from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Fase 1, primera pasada: registra odometria y observaciones de ArUco a
    CSV mientras se reproduce el rosbag de mapeo (tpf/rosbags/laberinto).

    Deliberadamente NO incluye corrected_map_node — ese nodo necesita las
    poses ya optimizadas por graph_slam.py (que se corre offline, despues
    de esta pasada), asi que correrlo ahora solo consumiria datos crudos
    sin sentido. La segunda pasada del bag (para construir el mapa) se
    hace con el ya existente tp_final.launch.py, sin cambios.
    """

    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="true al reproducir un rosbag, false en vivo.",
    )

    odom_logger = Node(
        package="tpf",
        executable="odom_logger",
        name="odom_logger",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    aruco_detector = Node(
        package="tpf",
        executable="aruco_detector",
        name="aruco_detector",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "save_csv": True,
            }
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        odom_logger,
        aruco_detector,
    ])
