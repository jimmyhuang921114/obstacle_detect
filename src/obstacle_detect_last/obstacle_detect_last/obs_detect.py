#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge
import torch
from ultralytics import YOLO
from collections import deque
from scipy.spatial import distance

class ObstacleDetectPublisher(Node):
    def __init__(self):
        super().__init__('obstacle_detect_publisher')
        
        # 初始化 CvBridge
        self.bridge = CvBridge()

        # 訂閱 RealSense 的彩色和深度圖像
        self.color_subscriber = self.create_subscription(
            Image, '/camera/color', self.color_callback, 10)
        self.depth_subscriber = self.create_subscription(
            Image, '/camera/depth', self.depth_callback, 10)

        # 發布物體的二維平面位置 (x: 歸一化位置, depth: 釐米)
        self.xy_publisher = self.create_publisher(
            Float32MultiArray, '/obstacle/xy_list', 10)

        # 加載 YOLOv8 分割模型
        self.model = YOLO("/home/darkdemon/ros2_ws/epoch90.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.get_logger().info("YOLOv8 model loaded successfully.")

        # 存儲彩色和深度圖像
        self.color_image = None
        self.depth_image = None

        # 內參數（需要根據 RealSense 相機校準數據填寫）
        self.fx = 640  # 焦距 x (像素)
        self.fy = 640  # 焦距 y (像素)
        self.cx = 320  # 光心 x (像素)
        self.cy = 240  # 光心 y (像素)

        # 歷史數據緩存
        self.tracked_masks = {}  # 格式: {track_id: {'history': deque, 'last_center': (x, y)}}
        self.next_track_id = 0
        self.iou_threshold = 0.3  # 遮罩匹配的 IoU 閾值
        self.max_disappeared = 5  # 最大允許消失的幀數
        self.history_size = 5  # 歷史數據緩存大小

    def color_callback(self, msg):
        """ 處理彩色圖像 """
        self.color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process_images()

    def depth_callback(self, msg):
        """ 處理深度圖像 """
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")
        self.process_images()

    def process_images(self):
        """ 處理彩色圖像與深度圖像，計算物體位置並顯示 """
        if self.color_image is None or self.depth_image is None:
            return

        # 調整圖像大小（可選）
        color_img = cv2.resize(self.color_image, (640, 480))
        depth_img = cv2.resize(self.depth_image, (640, 480))

        # 執行 YOLOv8 分割推理
        results = self.model(color_img)
        if len(results) == 0 or results[0].masks is None:
            self.get_logger().warn("No mask detected.")
            return

        # 複製彩色圖像並繪製半透明遮罩（藍色，透明度 30%）
        result_img = color_img.copy()
        for mask in results[0].masks.data:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_np, (640, 480))
            # 創建半透明遮罩 (BGR + Alpha)
            overlay = result_img.copy()
            overlay[mask_resized > 0] = (255, 0, 0)  # 藍色
            alpha = 0.3  # 透明度
            result_img = cv2.addWeighted(overlay, alpha, result_img, 1 - alpha, 0)

        # 計算物體的二維平面位置 (x: 歸一化, depth: 釐米)
        points = self.get_obstacle_points(depth_img, results[0].masks.data, step=20, max_points_per_mask=20)
        
        # 發布數據 (格式: [x1, depth1, x2, depth2, ...])
        xy_msg = Float32MultiArray()
        xy_msg.data = points.flatten().tolist()
        self.xy_publisher.publish(xy_msg)

        # 顯示帶遮罩的彩色圖像
        cv2.imshow("YOLOv8 Segmentation Result", result_img)
        cv2.waitKey(1)

    def get_obstacle_points(self, depth_image, masks, step=20, max_points_per_mask=20):
        """
        根據遮罩點位計算物體的二維平面位置。
        :param depth_image: 深度圖 (H, W)
        :param masks: YOLO 分割遮罩
        :param step: 取樣步長（每 step 個像素取一個點）
        :param max_points_per_mask: 每個遮罩的最大點數
        :return: (N,2) 數組，每行格式為 [歸一化x, 深度(cm)]
        """
        h, w = depth_image.shape
        points = []

        # 1. 提取當前幀的遮罩特徵
        current_masks = []
        for mask in masks:
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            mask_resized = cv2.resize(mask_np, (w, h))
            y_indices, x_indices = np.where(mask_resized > 0)
            if len(y_indices) == 0:
                continue
            center = (int(np.mean(x_indices)), int(np.mean(y_indices)))  # 座標轉換為整數
            current_masks.append({
                'mask': mask_resized,
                'center': center,
                'points': []
            })

        # 2. 初始化匹配標記陣列
        matched_current = [False] * len(current_masks)
        matched_tracks = {}

        # 3. 遮罩匹配 (基於中心點距離 + IoU)
        for track_id, track_data in self.tracked_masks.items():
            best_match = None
            min_distance = float('inf')
            best_iou = 0.0

            for idx, curr_mask in enumerate(current_masks):
                if matched_current[idx]:
                    continue

                # 計算中心點距離
                dist = distance.euclidean(track_data['last_center'], curr_mask['center'])
                
                # 計算 IoU
                intersection = np.logical_and(track_data['last_mask'], curr_mask['mask'])
                union = np.logical_or(track_data['last_mask'], curr_mask['mask'])
                iou = np.sum(intersection) / (np.sum(union) + 1e-6)  # 避免除以零

                if dist < 50 and iou > self.iou_threshold:
                    if iou > best_iou or (iou == best_iou and dist < min_distance):
                        best_match = idx
                        min_distance = dist
                        best_iou = iou

            if best_match is not None:
                matched_current[best_match] = True
                matched_tracks[track_id] = best_match

        # 4. 更新現有追蹤
        for track_id, curr_idx in matched_tracks.items():
            curr_mask = current_masks[curr_idx]
            self.tracked_masks[track_id]['last_center'] = curr_mask['center']
            self.tracked_masks[track_id]['last_mask'] = curr_mask['mask']
            self.tracked_masks[track_id]['disappeared'] = 0  # 重置消失計數

        # 5. 處理消失的追蹤
        for track_id in list(self.tracked_masks.keys()):
            if track_id not in matched_tracks:
                self.tracked_masks[track_id]['disappeared'] += 1
                if self.tracked_masks[track_id]['disappeared'] > self.max_disappeared:
                    del self.tracked_masks[track_id]

        # 6. 新增未匹配的遮罩
        for idx, curr_mask in enumerate(current_masks):
            if not matched_current[idx]:
                self.tracked_masks[self.next_track_id] = {
                    'history': deque(maxlen=self.history_size),
                    'last_center': curr_mask['center'],
                    'last_mask': curr_mask['mask'],
                    'disappeared': 0
                }
                self.next_track_id += 1

        # 7. 處理每個追蹤的遮罩
        for track_id, track_data in self.tracked_masks.items():
            mask_resized = track_data['last_mask']
            y_indices, x_indices = np.where(mask_resized > 0)
            
            # 採樣點
            mask_points = []
            for i in range(0, len(y_indices), step):
                if len(mask_points) >= max_points_per_mask:
                    break
                y = y_indices[i]
                x = x_indices[i]
                depth_mm = depth_image[y, x]
                if depth_mm == 0 or depth_mm > 5000:
                    continue
                x_normalized = (x / w) * 2 - 1
                depth_cm = depth_mm / 10.0
                mask_points.append([x_normalized, depth_cm])

            # 動態平滑權重 (移動越快，平滑越低)
            if len(track_data['history']) > 0:
                prev_center = track_data['history'][-1]['center']
                curr_center = track_data['last_center']
                movement = distance.euclidean(prev_center, curr_center)
                dynamic_weight = max(0.3, 0.7 - movement/100)  # 根據移動速度調整權重
                
                # 確保歷史點數與當前點數一致
                if len(track_data['history'][-1]['points']) == len(mask_points):
                    smoothed_points = []
                    for (x_prev, y_prev), (x_curr, y_curr) in zip(
                        track_data['history'][-1]['points'], mask_points):
                        x_smooth = dynamic_weight * x_prev + (1 - dynamic_weight) * x_curr
                        y_smooth = dynamic_weight * y_prev + (1 - dynamic_weight) * y_curr
                        smoothed_points.append([x_smooth, y_smooth])
                    mask_points = smoothed_points

            # 更新歷史
            track_data['history'].append({
                'points': mask_points,
                'center': track_data['last_center']
            })
            points.extend(mask_points)

        return np.array(points) if points else np.empty((0,2))

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()