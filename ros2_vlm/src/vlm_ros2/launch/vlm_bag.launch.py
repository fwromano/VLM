from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    # Declare launch arguments
    bag_file_arg = DeclareLaunchArgument(
        'bag_file',
        default_value='',
        description='Path to ROS2 bag file'
    )
    
    model_name_arg = DeclareLaunchArgument(
        'model_name',
        default_value='InternVL3-2B-hf',
        description='VLM model name'
    )
    
    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to run model on (cuda/cpu)'
    )
    
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/image_raw',
        description='Image topic from bag file'
    )
    
    analysis_rate_arg = DeclareLaunchArgument(
        'analysis_rate',
        default_value='1.0',
        description='Analysis rate in Hz'
    )
    
    # Play bag file
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_file'), '--loop'],
        output='screen'
    )
    
    # VLM node
    vlm_node = Node(
        package='vlm_ros2',
        executable='vlm_node.py',
        name='vlm',
        output='screen',
        parameters=[{
            'model_name': LaunchConfiguration('model_name'),
            'device': LaunchConfiguration('device'),
            'image_topic': LaunchConfiguration('image_topic'),
            'continuous_analysis': True,
            'analysis_rate': LaunchConfiguration('analysis_rate'),
            'default_prompt': 'Describe what you see in this image',
            'queue_size': 10,
        }]
    )
    
    # Echo node to display analysis results
    echo_node = ExecuteProcess(
        cmd=['ros2', 'topic', 'echo', '/vlm/analysis', 'vlm_ros2/msg/VLMAnalysis'],
        output='screen'
    )
    
    return LaunchDescription([
        bag_file_arg,
        model_name_arg,
        device_arg,
        image_topic_arg,
        analysis_rate_arg,
        bag_play,
        vlm_node,
        echo_node,
    ])