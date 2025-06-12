#!/usr/bin/env python3
"""
ROS2 VLM Node - Hybrid Approach
Uses system Python for ROS2, calls conda VLM via subprocess
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import subprocess
import json
import tempfile
import cv2
import os
import time
from pathlib import Path

class HybridVLMNode(Node):
    def __init__(self):
        super().__init__('hybrid_vlm_node')
        
        # ROS2 components
        self.bridge = CvBridge()
        
        # Publishers and subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.analysis_pub = self.create_publisher(String, '/vlm/analysis', 10)
        
        # VLM processing setup
        self.conda_env_path = "/home/fwromano/anaconda3/envs/vlm"
        self.vlm_script_path = str(Path(__file__).parent / "vlm_processor.py")
        
        # Processing state
        self.current_prompt = "What do you see in this image?"
        self.processing = False
        self.last_analysis_time = 0
        self.analysis_interval = 1.5  # Analyze every 1.5 seconds
        
        # Create subscription to change prompt
        self.prompt_sub = self.create_subscription(
            String, '/vlm/set_prompt', self.prompt_callback, 10)
        
        self.get_logger().info("Hybrid VLM Node started")
        self.get_logger().info(f"Using conda env: {self.conda_env_path}")
        self.get_logger().info(f"VLM script: {self.vlm_script_path}")
        self.get_logger().info(f"Analysis interval: {self.analysis_interval}s")
        self.get_logger().info(f"Current prompt: '{self.current_prompt}'")
    
    def prompt_callback(self, msg):
        """Topic callback to change the analysis prompt"""
        self.current_prompt = msg.data
        self.get_logger().info(f"Prompt updated to: '{self.current_prompt}'")
    
    def image_callback(self, msg):
        """Process incoming images"""
        current_time = time.time()
        
        # Check if enough time has passed since last analysis
        if self.processing or (current_time - self.last_analysis_time) < self.analysis_interval:
            return
            
        self.processing = True
        self.last_analysis_time = current_time
        
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Process with VLM
            result = self.process_with_vlm(cv_image, self.current_prompt)
            
            # Publish result
            result_msg = String()
            result_msg.data = result
            self.analysis_pub.publish(result_msg)
            
            self.get_logger().info(f"VLM result: {result}")
            
        except Exception as e:
            self.get_logger().error(f"Image processing failed: {e}")
        finally:
            self.processing = False
    
    def process_with_vlm(self, image, prompt):
        """Process image with VLM using conda environment"""
        try:
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                temp_image_path = tmp_file.name
            
            # Prepare command to run VLM in conda environment
            cmd = [
                f"{self.conda_env_path}/bin/python",
                self.vlm_script_path,
                temp_image_path,
                prompt
            ]
            
            # Run VLM processing
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"}
            )
            
            # Clean up temp file
            os.unlink(temp_image_path)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                error_msg = f"VLM process failed: {result.stderr}"
                self.get_logger().error(error_msg)
                return f"Error: {error_msg}"
                
        except subprocess.TimeoutExpired:
            self.get_logger().error("VLM processing timed out")
            return "Error: VLM processing timed out"
        except Exception as e:
            self.get_logger().error(f"VLM processing error: {e}")
            return f"Error: {e}"

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = HybridVLMNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()