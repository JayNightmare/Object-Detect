"""
Utility functions for the object detection application.

This module contains helper functions for image processing, file handling,
and other utility operations.
"""

import cv2
import numpy as np
import os
import time
from typing import List, Tuple, Optional
import urllib.request
import zipfile

def create_directories():
    """Create necessary directories for the application."""
    directories = ['screenshots', 'output', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def download_yolo_files():
    """
    Download YOLO configuration and weights files.
    
    Note: This is a placeholder function. In practice, you would download
    the actual YOLO files from the official repository.
    """
    print("YOLO file download functionality:")
    print("1. Download yolov3.weights from: https://pjreddie.com/media/files/yolov3.weights")
    print("2. Download yolov3.cfg from: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg")
    print("3. Download coco.names from: https://github.com/pjreddie/darknet/blob/master/data/coco.names")
    print("4. Place these files in the 'models' directory")

def resize_frame(frame: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Resize frame by scale factor.
    
    Args:
        frame: Input frame
        scale_factor: Scale factor (1.0 = original size, 0.5 = half size)
        
    Returns:
        Resized frame
    """
    if scale_factor == 1.0:
        return frame
    
    height, width = frame.shape[:2]
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    return cv2.resize(frame, (new_width, new_height))

def apply_gaussian_blur(frame: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.
    
    Args:
        frame: Input frame
        kernel_size: Size of the Gaussian kernel (must be odd)
        
    Returns:
        Blurred frame
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

def enhance_contrast(frame: np.ndarray, alpha: float = 1.2, beta: int = 10) -> np.ndarray:
    """
    Enhance frame contrast and brightness.
    
    Args:
        frame: Input frame
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Enhanced frame
    """
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

def draw_crosshair(frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 1) -> np.ndarray:
    """
    Draw crosshair in the center of the frame.
    
    Args:
        frame: Input frame
        color: Color of the crosshair (BGR)
        thickness: Line thickness
        
    Returns:
        Frame with crosshair
    """
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    
    # Draw horizontal line
    cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), color, thickness)
    # Draw vertical line
    cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), color, thickness)
    
    return frame

def save_detection_info(detections: List[dict], filename: Optional[str] = None):
    """
    Save detection information to a text file.
    
    Args:
        detections: List of detection dictionaries
        filename: Output filename (auto-generated if None)
    """
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detections_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"Detection Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        for i, detection in enumerate(detections):
            f.write(f"Detection {i + 1}:\n")
            f.write(f"  Class: {detection.get('class', 'Unknown')}\n")
            f.write(f"  Confidence: {detection.get('confidence', 0):.2f}\n")
            f.write(f"  Bounding Box: {detection.get('bbox', 'N/A')}\n")
            f.write("\n")
    
    print(f"Detection info saved to: {filename}")

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box [x, y, w, h]
        box2: Second bounding box [x, y, w, h]
        
    Returns:
        IoU value
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection area
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

def get_available_cameras() -> List[int]:
    """
    Get list of available camera indices.
    
    Returns:
        List of available camera indices
    """
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    return available_cameras

def format_detection_text(class_name: str, confidence: float) -> str:
    """
    Format detection text for display.
    
    Args:
        class_name: Name of detected class
        confidence: Detection confidence
        
    Returns:
        Formatted text string
    """
    return f"{class_name}: {confidence:.2f}"

def create_color_palette(num_classes: int) -> List[Tuple[int, int, int]]:
    """
    Create a color palette for different classes.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        List of BGR color tuples
    """
    colors = []
    for i in range(num_classes):
        hue = i * 180 // num_classes
        color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        bgr_color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr_color)))
    
    return colors

class FrameBuffer:
    """Simple frame buffer for storing recent frames."""
    
    def __init__(self, max_size: int = 30):
        """
        Initialize frame buffer.
        
        Args:
            max_size: Maximum number of frames to store
        """
        self.max_size = max_size
        self.frames = []
        self.timestamps = []
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to buffer."""
        self.frames.append(frame.copy())
        self.timestamps.append(time.time())
        
        # Remove oldest frames if buffer is full
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
            self.timestamps.pop(0)
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame."""
        return self.frames[-1] if self.frames else None
    
    def get_frame_at_index(self, index: int) -> Optional[np.ndarray]:
        """Get frame at specific index."""
        return self.frames[index] if 0 <= index < len(self.frames) else None
    
    def clear(self):
        """Clear the buffer."""
        self.frames.clear()
        self.timestamps.clear()
