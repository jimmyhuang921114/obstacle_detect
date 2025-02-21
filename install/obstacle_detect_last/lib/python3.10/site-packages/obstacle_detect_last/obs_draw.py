import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

class ObstaclePlotter(Node):
    def __init__(self):
        super().__init__('obstacle_plotter')

        # 订阅障碍物坐标
        self.subscription = self.create_subscription(
            Float32MultiArray, '/obstacle/xy_list', self.obstacle_callback, 10)

        # 维护坐标缓冲区（滑动窗口）
        self.buffer_size = 5
        self.obstacle_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()

        # 初始化 Matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7.2, 5.4), dpi=100)
        plt.show(block=False)

        # 定时更新绘图
        self.timer = self.create_timer(0.1, self.plot_obstacles)

    def obstacle_callback(self, msg):
        """ 处理接收到的坐标数据 """
        with self.lock:
            try:
                # 转换为 (N,2) 数组: [x, depth]
                data = np.array(msg.data).reshape(-1, 2)
                self.obstacle_buffer.append(data)
            except Exception as e:
                self.get_logger().error(f"数据解析失败: {e}")

    def get_smoothed_obstacles(self):
        """ 计算滑动窗口平均 """
        with self.lock:
            if len(self.obstacle_buffer) == 0:
                return np.array([])

            # 检查数据形状一致性
            shapes = [arr.shape for arr in self.obstacle_buffer]
            if not all(s == shapes[0] for s in shapes):
                return self.obstacle_buffer[-1]

            return np.mean(np.array(self.obstacle_buffer), axis=0)

    def plot_obstacles(self):
        """ 绘制二维平面图 """
        smoothed = self.get_smoothed_obstacles()

        self.ax.clear()
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(0, 500)  # Y轴范围 (cm)
        self.ax.set_xlabel("Normalized X")
        self.ax.set_ylabel("Depth (cm)")
        self.ax.set_title("Obstacle Map")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        if smoothed.size > 0:
            x = smoothed[:, 0]
            y = smoothed[:, 1]
            self.ax.scatter(x, y, c='red', edgecolors='black', label="Obstacles")
            self.ax.legend()
        else:
            self.ax.text(0, 250, "No Obstacles", ha='center', va='center', color='gray')

        self.fig.canvas.draw_idle()
        plt.pause(0.001)

def main(args=None):
    rclpy.init(args=args)
    node = ObstaclePlotter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()