#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, RegionOfInterest
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import time
import threading
from queue import Queue, Empty
from transformers import AutoProcessor, InternVLChat2Model
import gc

# Import custom messages
from vlm_ros2.msg import VLMAnalysis
from vlm_ros2.srv import AnalyzeImage, SetPrompt
from vlm_ros2.action import AnalyzeVideo


class VLMNode(Node):
    def __init__(self):
        super().__init__('vlm_node')
        
        # Declare parameters
        self.declare_parameter('model_name', 'InternVL3-2B-hf')
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('continuous_analysis', False)
        self.declare_parameter('analysis_rate', 1.0)  # Hz
        self.declare_parameter('default_prompt', 'What do you see in this image?')
        self.declare_parameter('queue_size', 10)
        
        # Get parameters
        self.model_name = self.get_parameter('model_name').value
        self.device = self.get_parameter('device').value
        self.image_topic = self.get_parameter('image_topic').value
        self.continuous_analysis = self.get_parameter('continuous_analysis').value
        self.analysis_rate = self.get_parameter('analysis_rate').value
        self.default_prompt = self.get_parameter('default_prompt').value
        self.queue_size = self.get_parameter('queue_size').value
        
        self.get_logger().info(f'Initializing VLM Node with model: {self.model_name}')
        self.get_logger().info(f'Using device: {self.device}')
        
        # Initialize model
        self.model = None
        self.processor = None
        self.model_lock = threading.Lock()
        self.init_model()
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Analysis queue
        self.analysis_queue = Queue(maxsize=self.queue_size)
        self.latest_image = None
        self.image_lock = threading.Lock()
        
        # Callback groups for concurrent execution
        self.cb_group = ReentrantCallbackGroup()
        
        # Publishers
        self.analysis_pub = self.create_publisher(
            VLMAnalysis, '/vlm/analysis', 10)
        self.status_pub = self.create_publisher(
            DiagnosticStatus, '/vlm/status', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10,
            callback_group=self.cb_group)
        
        # Services
        self.analyze_srv = self.create_service(
            AnalyzeImage, '/vlm/analyze_image', self.analyze_image_callback,
            callback_group=self.cb_group)
        self.set_prompt_srv = self.create_service(
            SetPrompt, '/vlm/set_prompt', self.set_prompt_callback,
            callback_group=self.cb_group)
        
        # Action server
        self.analyze_action = ActionServer(
            self, AnalyzeVideo, '/vlm/analyze_video',
            self.analyze_video_callback, callback_group=self.cb_group)
        
        # Timers
        if self.continuous_analysis:
            period = 1.0 / self.analysis_rate
            self.analysis_timer = self.create_timer(
                period, self.continuous_analysis_callback,
                callback_group=self.cb_group)
        
        # Status timer
        self.status_timer = self.create_timer(
            1.0, self.publish_status, callback_group=self.cb_group)
        
        # Processing thread
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.get_logger().info('VLM Node initialized successfully')
    
    def init_model(self):
        """Initialize the VLM model"""
        try:
            self.get_logger().info('Loading VLM model...')
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                f"lmms-lab/{self.model_name}",
                device_map=self.device
            )
            
            if self.device == 'cuda':
                self.model = InternVLChat2Model.from_pretrained(
                    f"lmms-lab/{self.model_name}",
                    torch_dtype=torch.bfloat16,
                    device_map=self.device
                )
            else:
                self.model = InternVLChat2Model.from_pretrained(
                    f"lmms-lab/{self.model_name}",
                    device_map=self.device
                )
            
            self.model.eval()
            self.get_logger().info('Model loaded successfully')
            
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise
    
    def image_callback(self, msg):
        """Store latest image for continuous analysis"""
        with self.image_lock:
            self.latest_image = msg
    
    def continuous_analysis_callback(self):
        """Timer callback for continuous analysis"""
        if self.latest_image is not None:
            with self.image_lock:
                image_msg = self.latest_image
            
            # Add to queue if not full
            if not self.analysis_queue.full():
                self.analysis_queue.put({
                    'image': image_msg,
                    'prompt': self.default_prompt,
                    'callback': self.publish_analysis
                })
    
    def analyze_image_callback(self, request, response):
        """Service callback for single image analysis"""
        try:
            # Process the image
            result = self.analyze_image_sync(request.image, request.prompt, request.roi)
            
            # Fill response
            response.analysis = result
            response.success = True
            response.error_message = ''
            
        except Exception as e:
            self.get_logger().error(f'Analysis failed: {str(e)}')
            response.success = False
            response.error_message = str(e)
        
        return response
    
    def set_prompt_callback(self, request, response):
        """Service callback to set default prompt and continuous mode"""
        self.default_prompt = request.prompt
        self.continuous_analysis = request.enable_continuous
        self.analysis_rate = request.analysis_rate
        
        # Update timer
        if hasattr(self, 'analysis_timer'):
            self.analysis_timer.cancel()
        
        if self.continuous_analysis and self.analysis_rate > 0:
            period = 1.0 / self.analysis_rate
            self.analysis_timer = self.create_timer(
                period, self.continuous_analysis_callback,
                callback_group=self.cb_group)
        
        response.success = True
        response.current_prompt = self.default_prompt
        response.continuous_enabled = self.continuous_analysis
        response.current_rate = self.analysis_rate
        
        return response
    
    def analyze_video_callback(self, goal_handle):
        """Action server callback for video analysis"""
        self.get_logger().info('Starting video analysis action')
        
        feedback_msg = AnalyzeVideo.Feedback()
        result = AnalyzeVideo.Result()
        
        start_time = time.time()
        analyses = []
        frames_analyzed = 0
        
        # Calculate analysis interval
        analysis_interval = 1.0 / goal_handle.request.analysis_rate
        last_analysis_time = 0
        
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if we should stop
            if goal_handle.request.duration_seconds > 0 and elapsed >= goal_handle.request.duration_seconds:
                break
            
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Video analysis cancelled')
                return AnalyzeVideo.Result()
            
            # Check if it's time for next analysis
            if current_time - last_analysis_time >= analysis_interval:
                with self.image_lock:
                    if self.latest_image is not None:
                        image_msg = self.latest_image
                        
                        # Analyze the image
                        analysis = self.analyze_image_sync(
                            image_msg, goal_handle.request.prompt)
                        analyses.append(analysis)
                        frames_analyzed += 1
                        
                        # Send feedback
                        feedback_msg.current_analysis = analysis
                        feedback_msg.frames_analyzed = frames_analyzed
                        feedback_msg.elapsed_time = elapsed
                        goal_handle.publish_feedback(feedback_msg)
                        
                        last_analysis_time = current_time
            
            time.sleep(0.01)  # Small sleep to prevent CPU spinning
        
        # Prepare result
        result.analyses = analyses
        result.total_frames_analyzed = frames_analyzed
        result.total_duration = time.time() - start_time
        
        # Generate summary
        if analyses:
            result.summary = f"Analyzed {frames_analyzed} frames over {result.total_duration:.1f} seconds. "
            result.summary += f"Common observations: {self.summarize_analyses(analyses)}"
        
        goal_handle.succeed()
        return result
    
    def analyze_image_sync(self, image_msg, prompt, roi=None):
        """Synchronous image analysis"""
        start_time = time.time()
        
        # Convert ROS image to OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(image_msg, "rgb8")
        
        # Apply ROI if provided
        if roi and roi.width > 0 and roi.height > 0:
            cv_image = cv_image[roi.y_offset:roi.y_offset+roi.height,
                               roi.x_offset:roi.x_offset+roi.width]
        
        # Perform VLM analysis
        with self.model_lock:
            analysis_text = self.run_vlm_inference(cv_image, prompt)
        
        # Create analysis message
        analysis = VLMAnalysis()
        analysis.header.stamp = self.get_clock().now().to_msg()
        analysis.header.frame_id = image_msg.header.frame_id
        analysis.text = analysis_text
        analysis.prompt = prompt
        analysis.confidence = 1.0  # Could implement confidence scoring
        analysis.processing_time = time.time() - start_time
        analysis.model_name = self.model_name
        if roi:
            analysis.roi = roi
        
        return analysis
    
    def run_vlm_inference(self, image, prompt):
        """Run VLM inference on image"""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=0.7
                )
            
            # Decode response
            response = self.processor.decode(
                outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response
            if prompt in response:
                response = response.split(prompt)[-1].strip()
            
            return response
            
        except Exception as e:
            self.get_logger().error(f'Inference error: {str(e)}')
            return f"Error during analysis: {str(e)}"
    
    def process_queue(self):
        """Background thread to process analysis queue"""
        while rclpy.ok():
            try:
                task = self.analysis_queue.get(timeout=1.0)
                
                # Perform analysis
                analysis = self.analyze_image_sync(
                    task['image'], task['prompt'])
                
                # Call the callback
                if task['callback']:
                    task['callback'](analysis)
                    
            except Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Queue processing error: {str(e)}')
    
    def publish_analysis(self, analysis):
        """Publish analysis result"""
        self.analysis_pub.publish(analysis)
    
    def publish_status(self):
        """Publish node status"""
        status = DiagnosticStatus()
        status.level = DiagnosticStatus.OK
        status.name = 'VLM Node'
        status.message = 'Operating normally'
        status.hardware_id = self.device
        
        # Add key values
        status.values.append(KeyValue(key='model', value=self.model_name))
        status.values.append(KeyValue(key='device', value=self.device))
        status.values.append(KeyValue(key='continuous_mode', 
                                    value=str(self.continuous_analysis)))
        status.values.append(KeyValue(key='queue_size', 
                                    value=f"{self.analysis_queue.qsize()}/{self.queue_size}"))
        
        self.status_pub.publish(status)
    
    def summarize_analyses(self, analyses):
        """Create a summary of multiple analyses"""
        # Simple implementation - could be enhanced with NLP
        texts = [a.text for a in analyses[-5:]]  # Last 5 analyses
        return ' '.join(texts[:2])  # Return first 2 for brevity


def main(args=None):
    rclpy.init(args=args)
    
    node = VLMNode()
    
    # Use MultiThreadedExecutor for concurrent callbacks
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()