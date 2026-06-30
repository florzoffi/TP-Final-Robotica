from launch import LaunchDescription
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    rviz_config = os.path.join(
        get_package_share_directory("tpf"),
        "rviz",
        "tpf.rviz",
    )

    aruco_detector = Node(
        package="tpf",
        executable="aruco_detector",
        parameters=[
            {
                "use_sim_time": True,
                "save_csv": False
            }
        ]
    )

    corrected_map_node = Node(
        package="tpf",
        executable="corrected_map_node",
        name="corrected_map_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    landmarks_publisher = Node(
        package="tpf",
        executable="landmarks_publisher",
        name="landmarks_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([
        aruco_detector,
        corrected_map_node,
        landmarks_publisher,
        rviz,
    ])