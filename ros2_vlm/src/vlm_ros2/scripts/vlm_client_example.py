#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from vlm_ros2.msg import VLMAnalysis
from vlm_ros2.srv import AnalyzeImage, SetPrompt
from vlm_ros2.action import AnalyzeVideo
import sys


class VLMClientExample(Node):
    def __init__(self):
        super().__init__('vlm_client_example')
        
        # Subscribers
        self.analysis_sub = self.create_subscription(
            VLMAnalysis, '/vlm/analysis', self.analysis_callback, 10)
        
        # Service clients
        self.analyze_client = self.create_client(AnalyzeImage, '/vlm/analyze_image')
        self.set_prompt_client = self.create_client(SetPrompt, '/vlm/set_prompt')
        
        # Action client
        self.analyze_video_client = ActionClient(self, AnalyzeVideo, '/vlm/analyze_video')
        
        self.get_logger().info('VLM Client Example initialized')
    
    def analysis_callback(self, msg):
        """Callback for continuous analysis results"""
        self.get_logger().info(f'Analysis received:')
        self.get_logger().info(f'  Prompt: {msg.prompt}')
        self.get_logger().info(f'  Response: {msg.text}')
        self.get_logger().info(f'  Processing time: {msg.processing_time:.2f}s')
    
    def set_continuous_mode(self, prompt, rate=1.0):
        """Enable continuous analysis mode"""
        if not self.set_prompt_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Set prompt service not available')
            return
        
        request = SetPrompt.Request()
        request.prompt = prompt
        request.enable_continuous = True
        request.analysis_rate = rate
        
        future = self.set_prompt_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.success:
            self.get_logger().info(f'Continuous mode enabled: {response.current_prompt} @ {response.current_rate}Hz')
        else:
            self.get_logger().error('Failed to enable continuous mode')
    
    def analyze_single_image(self, prompt):
        """Request analysis of current camera image"""
        if not self.analyze_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Analyze service not available')
            return
        
        # Get latest image from camera topic
        msg = self.get_latest_image()
        if msg is None:
            self.get_logger().error('No image available')
            return
        
        request = AnalyzeImage.Request()
        request.image = msg
        request.prompt = prompt
        
        future = self.analyze_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        if response.success:
            self.get_logger().info(f'Analysis result: {response.analysis.text}')
        else:
            self.get_logger().error(f'Analysis failed: {response.error_message}')
    
    def analyze_video_stream(self, prompt, duration=10.0, rate=0.5):
        """Analyze video stream for a duration"""
        if not self.analyze_video_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Analyze video action not available')
            return
        
        goal = AnalyzeVideo.Goal()
        goal.prompt = prompt
        goal.duration_seconds = duration
        goal.analysis_rate = rate
        
        self.get_logger().info(f'Starting video analysis for {duration}s at {rate}Hz')
        
        send_goal_future = self.analyze_video_client.send_goal_async(
            goal, feedback_callback=self.video_feedback_callback)
        
        rclpy.spin_until_future_complete(self, send_goal_future)
        goal_handle = send_goal_future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error('Video analysis goal rejected')
            return
        
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        self.get_logger().info(f'Video analysis complete:')
        self.get_logger().info(f'  Frames analyzed: {result.total_frames_analyzed}')
        self.get_logger().info(f'  Duration: {result.total_duration:.1f}s')
        self.get_logger().info(f'  Summary: {result.summary}')
    
    def video_feedback_callback(self, feedback_msg):
        """Callback for video analysis feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Frame {feedback.frames_analyzed}: {feedback.current_analysis.text[:100]}...')
    
    def get_latest_image(self):
        """Get latest image from camera topic"""
        try:
            msg = self.create_subscription(
                Image, '/camera/image_raw', lambda m: m, 1)
            rclpy.spin_once(self, timeout_sec=1.0)
            return msg
        except:
            return None


def main(args=None):
    rclpy.init(args=args)
    
    client = VLMClientExample()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'continuous':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "What do you see?"
            rate = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
            client.set_continuous_mode(prompt, rate)
            rclpy.spin(client)
            
        elif command == 'single':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "Describe this image"
            client.analyze_single_image(prompt)
            
        elif command == 'video':
            prompt = sys.argv[2] if len(sys.argv) > 2 else "What is happening?"
            duration = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
            rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
            client.analyze_video_stream(prompt, duration, rate)
            
        else:
            print("Usage: vlm_client_example.py [continuous|single|video] [prompt] [rate/duration]")
    else:
        # Default: enable continuous mode
        client.set_continuous_mode("What do you see in this image?", 0.5)
        rclpy.spin(client)
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()