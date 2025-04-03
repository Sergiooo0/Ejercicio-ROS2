from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='environment',
            executable='env',
            name='environment',
            output='screen'
        ),
        Node(
            package='environment',
            executable='car',
            name='car',
            output='screen'
        ),
        Node(
            package='controller_car',
            executable='controller_car',
            name='controller_car',
            output='screen'
        )
    ])
