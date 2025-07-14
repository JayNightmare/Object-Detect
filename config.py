"""
Configuration file for object detection settings.

This file contains configurable parameters for the object detection application.
You can modify these settings to customize the behavior of the detector.
Following Azure best practices for configuration management.
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.6

# Model settings
MODEL_TYPE = "yolov8"  # Options: "opencv_dnn", "yolov3", "yolov5", "yolov8", "tensorflow", "pytorch"

# YOLOv8 Ultralytics settings
YOLO_MODEL = "yolov8n.pt"  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
YOLO_DEVICE = "cpu"  # Options: "cpu", "cuda", "mps" (for Mac M1/M2)
YOLO_IMGSZ = 1280  # Input image size for YOLO model
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

# Object tracking configuration (following Azure best practices)
TRACKING_ENABLED = True
IMPORTANT_OBJECTS = [
    "person", "car", "bicycle", "motorcycle", "bus", "truck", "backpack",
    "handbag", "suitcase", "laptop", "cell phone", "book", "bottle", 
    "cup", "knife", "spoon", "bowl", "chair", "dining table", "couch",
    "tv", "remote", "keyboard", "mouse", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]
TRACKING_MEMORY_DURATION = 300  # seconds to remember object locations
TRACKING_MIN_CONFIDENCE = 0.8  # minimum confidence to track an object
TRACKING_DISTANCE_THRESHOLD = 100  # pixels - objects closer than this are considered same instance
TRACKING_MAX_OBJECTS = 1000  # maximum number of objects to track simultaneously
SHOW_LAST_SEEN_INFO = True  # show tracking information on screen
TRACKING_HISTORY_FILE = "object_tracking_history.json"  # file to save tracking history
TRACKING_ENABLE_LOGGING = True  # enable structured logging for tracking
