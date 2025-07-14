"""
Configuration file for object detection settings.

This file contains configurable parameters for the object detection application.
You can modify these settings to customize the behavior of the detector.
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Model settings
MODEL_TYPE = "yolov8"  # Options: "opencv_dnn", "yolov3", "yolov5", "yolov8", "tensorflow", "pytorch"

# YOLOv8 Ultralytics settings
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
YOLO_DEVICE = "cpu"  # Options: "cpu", "cuda", "mps" (for Mac M1/M2)
YOLO_IMGSZ = 640  # Input image size for YOLO model
YOLO_HALF = False  # Use half precision (FP16) for faster inference
YOLO_VERBOSE = False  # Show detailed model loading info

# Legacy YOLO model paths (for manual model files)
YOLO_WEIGHTS = "models/yolov3.weights"
YOLO_CONFIG = "models/yolov3.cfg"
YOLO_CLASSES = "models/coco.names"

# Colors for different object classes (BGR format)
CLASS_COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (255, 0, 0),         # Blue
    'bicycle': (0, 0, 255),     # Red
    'motorcycle': (255, 255, 0), # Cyan
    'bus': (255, 0, 255),       # Magenta
    'truck': (0, 255, 255),     # Yellow
    'default': (128, 128, 128)  # Gray
}

# Display settings
SHOW_FPS = True
SHOW_CONFIDENCE = True
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Recording settings
ENABLE_RECORDING = False
OUTPUT_VIDEO_PATH = "output/detection_output.avi"
VIDEO_CODEC = "XVID"
VIDEO_FPS = 20.0

# Screenshot settings
SCREENSHOT_PATH = "screenshots/"
SCREENSHOT_FORMAT = "jpg"

# Performance settings
SKIP_FRAMES = 0  # Skip every N frames for better performance (0 = process all frames)
RESIZE_FACTOR = 1.0  # Resize input frames (1.0 = no resize, 0.5 = half size)

# Advanced detection settings
ENABLE_TRACKING = False  # Enable object tracking between frames
MAX_TRACKING_DISTANCE = 50  # Maximum distance for object tracking
MIN_DETECTION_SIZE = 30  # Minimum size of detections to display
