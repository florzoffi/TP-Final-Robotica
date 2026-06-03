from setuptools import find_packages, setup

package_name = 'tpf'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'obstacle_avoidance_tb4 = tpf.obstacle_avoidance_tb4:main',
            'aruco_detector = tpf.aruco_detector:main',
            'odom_logger = tpf.odom_logger:main',
        ],
    },
)
