from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():

    # Paquetes que vamos a usar
    pkg_gazebo_ros = get_package_share_directory("gazebo_ros")
    pkg_turtlebot3_gazebo = get_package_share_directory("turtlebot3_gazebo")
    pkg_simulation = get_package_share_directory("turtlebot3_custom_simulation")
    pkg_tpf = get_package_share_directory("tpf")
    map_yaml = os.path.join(
        pkg_tpf,
        "maps",
        "map.yaml",
    )

    # Reloj de simulación
    use_sim_time = LaunchConfiguration("use_sim_time", default="true")

    # Pose inicial del robot en Gazebo
    x_pose = LaunchConfiguration("x_pose", default="0.0")
    y_pose = LaunchConfiguration("y_pose", default="0.0")

    # Mundo sin obstáculos
    # world = os.path.join(pkg_simulation, "worlds", "casa.world")

    # Mundo de la casa con obstaculos
    world = os.path.join(
        pkg_simulation,
        "worlds",
        "casa_o.world",
    )

    # Para que Gazebo encuentre los modelos del mundo
    models_path = os.path.join(pkg_simulation, "worlds")
    os.environ["GAZEBO_MODEL_PATH"] = (
        models_path + ":" + os.environ.get("GAZEBO_MODEL_PATH", "")
    )

    # Gazebo server
    gzserver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzserver.launch.py")
        ),
        launch_arguments={"world": world}.items(),
    )

    # Gazebo client
    gzclient = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, "launch", "gzclient.launch.py")
        )
    )

    # Publica el modelo y los TF internos del TurtleBot
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_turtlebot3_gazebo,
                "launch",
                "robot_state_publisher.launch.py",
            )
        ),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )

    # Spawnea el TurtleBot3 dentro de Gazebo
    spawn_turtlebot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                pkg_turtlebot3_gazebo,
                "launch",
                "spawn_turtlebot3.launch.py",
            )
        ),
        launch_arguments={
            "x_pose": x_pose,
            "y_pose": y_pose,
        }.items(),
    )

    # TF temporal map -> odom.
    # Más adelante, cuando implementemos el localizador, esto se reemplaza.
    static_tf_map_to_odom = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_tf_map_to_odom",
        arguments=[
            "0", "0", "0",
            "0", "0", "0",
            "map", "odom",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    
    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"yaml_filename": map_yaml},
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map_server",
        output="screen",
        parameters=[
            {"use_sim_time": True},
            {"autostart": True},
            {"node_names": ["map_server"]},
        ],
    )

    # RViz con la configuración de Parte B
    rviz_config = os.path.join(
        pkg_tpf,
        "rviz",
        "parte_b.rviz",
    )

    rviz = TimerAction(
        period=5.0,
        actions=[
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": True}],
            )
        ],
    )

    particle_localizer = Node(
        package="tpf",
        executable="particle_localizer",
        name="particle_localizer",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    
    path_planner = Node(
        package="tpf",
        executable="path_planner",
        name="path_planner",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    
    path_follower = Node(
        package="tpf",
        executable="path_follower",
        name = "path_follower",
        output = "screen",
        parameters=[{"use_sim_tim": True}],
    )
    
    navigation_manager = Node(
        package="tpf",
        executable="navigation_manager",
        name = "navigation_manager",
        output = "screen",
        parameters=[{"use_sim_tim": True}],
    )
        
    return LaunchDescription([
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_turtlebot,
        static_tf_map_to_odom,
        map_server,
        lifecycle_manager,
        particle_localizer,
        path_planner,
        path_follower,
        navigation_manager,
        rviz,
    ])