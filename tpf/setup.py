from setuptools import find_packages, setup
from glob import glob

package_name = 'tpf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/tpf/launch', glob('launch/*.launch.py')),
        ('share/tpf/rviz', glob('rviz/*.rviz')),
        ('share/tpf/maps', glob('maps/*')),
        ('share/tpf/config', glob('config/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='florenciazoffi',
    maintainer_email='fzoffi@udesa.edu.ar',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'obstacle_avoidance = tpf.obstacle_avoidance:main',
            'aruco_detector = tpf.aruco_detector:main',
            'odom_logger = tpf.odom_logger:main',
            'analyze_logs = tpf.analyze_logs:main',
            'graph_slam = tpf.graph_slam:main',
            'scan_logger = tpf.scan_logger:main',
            'corrected_map_node = tpf.corrected_map_node:main',
            'landmarks_publisher = tpf.landmarks_publisher:main',
            'particle_localizer = tpf.particle_localizer:main',
            'path_planner = tpf.path_planner:main',
            'path_follower = tpf.path_follower:main',
            'navigation_manager = tpf.navigation_manager:main',
            'virtual_landmark_sensor = tpf.virtual_landmark_sensor:main',
            'cone_detector = tpf.cone_detector:main',
            'cone_mission_manager = tpf.cone_mission_manager:main',
        ],
    },
)
