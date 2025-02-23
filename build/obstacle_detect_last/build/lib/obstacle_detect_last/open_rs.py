import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from cv_bridge import CvBridge
import time
from datetime import datetime
import os
import argparse
from threading import Lock

class RealSenseManager(Node):
    def __init__(self):
        super().__init__('realsense_manager')
        self.bridge = CvBridge()
        self.lock = Lock()
        self.MAX_DEPTH_MM = 5000  # 基础深度范围设置
        
        # 初始化配置
        self._parse_arguments()
        self._init_mode()

    def _parse_arguments(self):
        """命令行参数解析"""
        parser = argparse.ArgumentParser(description='RealSense基础管理器')
        parser.add_argument('--live', action='store_true', 
                         help='启用实时模式（默认）')
        parser.add_argument('--playback', action='store_true',
                         help='启用回放模式')
        parser.add_argument('--rgb', type=str, default='',
                         help='RGB视频路径（回放模式必需）')
        parser.add_argument('--depth', type=str, default='',
                         help='深度视频路径（回放模式必需）')
        self.args = parser.parse_args()
        
        # 模式冲突检查
        if self.args.playback and (not self.args.rgb or not self.args.depth):
            self.get_logger().error("回放模式需要指定--rgb和--depth参数")
            raise ValueError("Missing playback files")

    def _init_mode(self):
        """模式初始化"""
        if self.args.playback:
            self._init_playback()
        else:
            self._init_realsense()

    def _init_realsense(self):
        """初始化RealSense硬件"""
        try:
            # 创建ROS发布器
            self.color_pub = self.create_publisher(Image, '/camera/color', 10)
            self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)
            
            # 配置基础流参数
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # 启动管道
            self.pipeline = rs.pipeline()
            self.pipeline.start(config)
            
            # 设置录制功能
            self._init_recording()
            
            # 启动定时器
            self.create_timer(1/30.0, self._realsense_callback)
            self.get_logger().info("实时模式已启动")

        except Exception as e:
            self.get_logger().error(f"硬件初始化失败: {str(e)}")
            raise

    def _init_playback(self):
        """视频回放初始化"""
        if not all(map(os.path.exists, [self.args.rgb, self.args.depth])):
            self.get_logger().error("视频文件不存在")
            raise FileNotFoundError("Missing video files")
        
        # 创建视频读取器
        self.rgb_reader = cv2.VideoCapture(self.args.rgb)
        self.depth_reader = cv2.VideoCapture(self.args.depth)
        
        # 获取并验证帧率
        fps = self.rgb_reader.get(cv2.CAP_PROP_FPS) or 30
        
        # 创建ROS发布器
        self.color_pub = self.create_publisher(Image, '/camera/color', 10)
        self.depth_pub = self.create_publisher(Image, '/camera/depth', 10)
        
        # 启动定时器
        self.create_timer(1/fps, self._playback_callback)
        self.get_logger().info(f"回放模式已启动 ({fps}fps)")

    def _init_recording(self):
        """初始化录制功能"""
        os.makedirs("recordings", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recorder = {
            'color': cv2.VideoWriter(f"recordings/color_{timestamp}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480)),
            'depth': cv2.VideoWriter(f"recordings/depth_{timestamp}.avi", cv2.VideoWriter_fourcc(*'XVID'), 30, (640,480))
        }

    def _realsense_callback(self):
        """实时数据回调"""
        with self.lock:
            try:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    return
                
                # 获取原始数据
                color_img = np.asanyarray(color_frame.get_data())
                depth_data = np.asanyarray(depth_frame.get_data())
                
                # 发布数据
                self.color_pub.publish(self.bridge.cv2_to_imgmsg(color_img, 'bgr8'))
                self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_data, '16UC1'))
                
                # 录制和显示
                self._record_and_show(color_img, depth_data)

            except Exception as e:
                self.get_logger().error(f"帧处理异常: {str(e)}")

    def _record_and_show(self, color, depth):
        """统一处理录制和显示"""
        # 转换深度可视化
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(depth, alpha=255/self.MAX_DEPTH_MM),
            cv2.COLORMAP_JET
        )
        
        # 录制
        self.recorder['color'].write(color)
        self.recorder['depth'].write(depth_vis)
        
        # 显示
        cv2.imshow("Color Preview", color)
        cv2.imshow("Depth View", depth_vis)
        if cv2.waitKey(1) == ord('q'):
            self.destroy_node()

    def _playback_callback(self):
        """回放数据处理"""
        with self.lock:
            ret_rgb, rgb = self.rgb_reader.read()
            ret_depth, depth = self.depth_reader.read()
            
            if not ret_rgb or not ret_depth:
                self.get_logger().info("视频回放结束")
                self._cleanup_playback()
                return
                
            # 转换深度数据
            depth_gray = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
            depth_data = (depth_gray * (self.MAX_DEPTH_MM/255)).astype(np.uint16)
            
            # 发布数据
            self.color_pub.publish(self.bridge.cv2_to_imgmsg(rgb, 'bgr8'))
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(depth_data, '16UC1'))
            
            # 显示回放
            cv2.imshow("Playback RGB", rgb)
            cv2.imshow("Playback Depth", depth)
            if cv2.waitKey(1) == ord('q'):
                self.destroy_node()

    def _cleanup_playback(self):
        """清理回放资源"""
        self.rgb_reader.release()
        self.depth_reader.release()
        cv2.destroyAllWindows()

    def destroy_node(self):
        """资源释放"""
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        if hasattr(self, 'recorder'):
            for w in self.recorder.values():
                w.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    try:
        node = RealSenseManager()
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户终止操作")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()