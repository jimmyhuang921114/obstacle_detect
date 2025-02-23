import cv2

cap = cv2.VideoCapture('/home/darkdemon/ros2_ws/output_videos/depth_video_2025-02-23_14-28-43.avi')
ret, frame = cap.read()
if ret:
    print("Shape:", frame.shape)  # (height, width, channels) or (height, width) 
    print("Data type:", frame.dtype)  # e.g. uint8, uint16, float32, ...
