from pathlib import Path

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

PACKAGE_DIR = Path.home() / 'ros2_ws' / 'src' / 'cockroach_biology'
URDF_FILE = PACKAGE_DIR / 'urdf' / 'cockroach.urdf'
PUBLISHER_FILE = PACKAGE_DIR / 'cockroach_biology' / 'csv_gait_publisher.py'


def generate_launch_description():

    robot_description = URDF_FILE.read_text()

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_description
        }]
    )

    gait_publisher = ExecuteProcess(
        cmd=['python3', str(PUBLISHER_FILE)],
        output='screen'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='cockroach_rviz',
        output='screen'
    )

    return LaunchDescription([
        robot_state_publisher,
        gait_publisher,
        rviz
    ])
