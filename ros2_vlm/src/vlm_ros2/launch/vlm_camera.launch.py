from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    # Declare launch arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='Camera device path'
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
    
    continuous_arg = DeclareLaunchArgument(
        'continuous_analysis',
        default_value='true',
        description='Enable continuous analysis'
    )
    
    analysis_rate_arg = DeclareLaunchArgument(
        'analysis_rate',
        default_value='0.5',
        description='Analysis rate in Hz'
    )
    
    # USB camera node
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'image_width': 1280,
            'image_height': 720,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'camera_optical_frame',
            'io_method': 'mmap',
            'framerate': 30.0,
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
        ]
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
            'image_topic': '/camera/image_raw',
            'continuous_analysis': LaunchConfiguration('continuous_analysis'),
            'analysis_rate': LaunchConfiguration('analysis_rate'),
            'default_prompt': 'What do you see in this image?',
            'queue_size': 10,
        }]
    )
    
    # Image view node (optional visualization)
    image_view_node = Node(
        package='image_view',
        executable='image_view',
        name='image_viewer',
        output='screen',
        remappings=[
            ('image', '/camera/image_raw'),
        ],
        parameters=[{
            'autosize': True,
        }]
    )
    
    return LaunchDescription([
        camera_device_arg,
        model_name_arg,
        device_arg,
        continuous_arg,
        analysis_rate_arg,
        usb_cam_node,
        vlm_node,
        image_view_node,
    ])