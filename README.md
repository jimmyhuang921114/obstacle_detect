# 基於 YOLOv8 和 RealSense 相機的障礙物檢測

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)]
[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10-green)]
[![YOLOv8](https://img.shields.io/badge/YOLOv8-物件偵測-orange)]
[![RealSense](https://img.shields.io/badge/RealSense-深度相機-red)]

這是一個基於 ROS2 的障礙物檢測套件，使用 YOLOv8 進行物件分割，並結合 Intel RealSense 相機的深度資訊，實現環境中障礙物的即時檢測與定位。

---

## 功能特色

- 即時障礙物檢測：使用 YOLOv8 進行精確的物件分割。
- 深度感知：整合 Intel RealSense 相機獲取深度資訊。
- 2D 位置計算：將深度數據轉換為 2D 座標（歸一化的 x 值和深度值，單位：公分）。
- 視覺化：提供即時繪圖功能，顯示檢測到的障礙物位置。
- 平滑追蹤：使用歷史數據平滑和動態權重，確保追蹤穩定性。

---

## 安裝步驟

### 環境需求

1. ROS2 Humble（或其他相容的 ROS2 版本）。
2. Intel RealSense SDK 和驅動程式。
3. Python 3.8+ 及相關依賴套件。

### 安裝流程

1. 克隆專案：

git clone https://github.com/jimmyhuang921114/obstacle_detect.git

cd obstacle_detect

2. 安裝 Python 依賴套件：

pip install -r requirements.txt

3. 編譯 ROS2 套件：

colcon build --packages-select obstacle_detect_last
source install/setup.bash

4. 下載 YOLOv8 模型：
- 將 YOLOv8 模型檔案（yolov8l-seg.pt）放置於指定目錄（例如：/home/ros2_ws/yolov8l-seg.pt）。

---

## 使用說明

### 啟動節點

1. 啟動 RealSense 發佈節點：

ros2 run obstacle_detect_last open_rs

2. 啟動障礙物檢測節點：

ros2 run obstacle_detect_last obs_detect

3. 啟動視覺化節點：

ros2 run obstacle_detect_last obs_draw
### 使用 Launch 檔案

您也可以使用提供的 Launch 檔案一次性啟動所有節點：

ros2 launch obstacle_detect_last obstacle_detection.launch.py

---

## 節點概述

### 1. realsense_publisher
- 從 RealSense 相機發佈彩色圖像和深度圖像。
- 發佈的 Topic：
  - /camera/color (sensor_msgs/Image)
  - /camera/depth (sensor_msgs/Image)

### 2. obstacle_detect
- 使用 YOLOv8 和深度數據進行障礙物檢測。
- 發佈障礙物的 2D 位置（歸一化的 x 值和深度值）。
- 發佈的 Topic：
  - /obstacle/xy_list (std_msgs/Float32MultiArray)

### 3. obstacle_plotter
- 在 2D 圖表中視覺化檢測到的障礙物。
- 圖表會即時更新。

---
