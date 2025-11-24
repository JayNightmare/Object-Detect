"""
Configuration file for object detection settings.

This file contains configurable parameters for the object detection application.
You can modify these settings to customize the behavior of the detector.
Following Azure best practices for configuration management.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESHOLD = 0.6

# Model settings
MODEL_TYPE = "yolov11"  # Options: "opencv_dnn", "yolov3", "yolov5", "yolov11", "tensorflow", "pytorch"

# YOLOv11 Ultralytics settings
YOLO_MODEL = (
    "yolo11n.pt"  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
)
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
    "person": (0, 255, 0),  # Green
    "car": (255, 0, 0),  # Blue
    "bicycle": (0, 0, 255),  # Red
    "motorcycle": (255, 255, 0),  # Cyan
    "bus": (255, 0, 255),  # Magenta
    "truck": (0, 255, 255),  # Yellow
    "default": (128, 128, 128),  # Gray
}

# Display settings
SHOW_FPS = True
SHOW_CONFIDENCE = True
FONT_SCALE = 0.5
FONT_THICKNESS = 1

# Recording settings
ENABLE_RECORDING = True
OUTPUT_VIDEO_PATH = "output/"
VIDEO_CODEC = "mp4v"
VIDEO_FPS = 30.0
VIDEO_FILENAME_PREFIX = "detection_output"

# Screenshot settings
SCREENSHOT_PATH = "screenshots/"
SCREENSHOT_FORMAT = "jpg"

# Performance settings
SKIP_FRAMES = 0  # Skip every N frames for better performance (0 = process all frames)
RESIZE_FACTOR = 1.0  # Resize input frames (1.0 = no resize, 0.5 = half size)

# Advanced detection settings
# Note: Tracking settings are now handled in the "Object tracking configuration" section below

# Object tracking configuration (following Azure best practices)
TRACKING_ENABLED = True
IMPORTANT_OBJECTS = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
    "laptop",
    "cell phone",
    "book",
    "bottle",
    "cup",
    "knife",
    "spoon",
    "bowl",
    "chair",
    "dining table",
    "couch",
    "tv",
    "remote",
    "keyboard",
    "mouse",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
TRACKING_MEMORY_DURATION = 300  # seconds to remember object locations
TRACKING_MIN_CONFIDENCE = 0.8  # minimum confidence to track an object
TRACKING_DISTANCE_THRESHOLD = (
    100  # pixels - objects closer than this are considered same instance
)
TRACKING_MAX_OBJECTS = 1000  # maximum number of objects to track simultaneously
SHOW_LAST_SEEN_INFO = True  # show tracking information on screen
TRACKING_HISTORY_FILE = "object_tracking_history.json"  # file to save tracking history
TRACKING_ENABLE_LOGGING = True  # enable structured logging for tracking

# AI Interpretation settings (OpenRouter)
AI_ENABLED = True
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # grab from .env file
OPENROUTER_MODEL = "x-ai/grok-4.1-fast"  # Model to use
AI_THOUGHT_INTERVAL = 10.0  # Seconds between passive "thoughts"
AI_SYSTEM_PROMPT = (
    "You are an AI assistant analyzing a video feed. "
    "You will be provided with a list of detected objects, their positions (zones), "
    "and confidence levels. "
    "Your goal is to interpret the scene, describe what is happening, "
    "and answer user questions based on this data. "
    "Keep your responses concise and relevant to the visual context."
)
