#!/usr/bin/env python3
"""
Camera Object Detection Application with YOLOv8

This application uses your computer's camera to detect objects in real-time
using YOLOv8 from Ultralytics.
"""

import cv2
import numpy as np
import argparse
import time
import os
from typing import List, Tuple, Optional
from pathlib import Path

# Import YOLOv8 from Ultralytics
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: Ultralytics YOLO not available. Install with: pip install ultralytics")

# # Import project config
# try:
#     # import config
# except ImportError:
print("Warning: config.py not found. Using default settings.")
class config:
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    MODEL_TYPE = "yolov8"
    YOLO_MODEL = "yolov8n.pt"
    YOLO_DEVICE = "cpu"
    YOLO_IMGSZ = 640
    YOLO_HALF = False
    YOLO_VERBOSE = False
    SHOW_FPS = True
    SHOW_CONFIDENCE = True
    SCREENSHOT_PATH = "screenshots/"
    SCREENSHOT_FORMAT = "jpg"

class YOLOv8Detector:
    """Real-time object detection using YOLOv8 from Ultralytics."""
    
    def __init__(self, 
                 model_name: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 device: str = "cpu",
                 imgsz: int = 640,
                 half: bool = False,
                 verbose: bool = False):
        """
        Initialize the YOLOv8 detector.
        
        Args:
            model_name: YOLOv8 model name (yolov8n.pt, yolov8s.pt, etc.)
            confidence_threshold: Minimum confidence for object detection
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            imgsz: Input image size
            half: Use half precision (FP16) for faster inference
            verbose: Show detailed model loading info
        """
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO is not available. Install with: pip install ultralytics")
        
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.verbose = verbose
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model: {model_name}")
        try:
            self.model = YOLO(model_name)
            if verbose:
                print(f"✅ Model loaded successfully: {model_name}")
                print(f"Device: {device}")
                print(f"Image size: {imgsz}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # COCO class names (YOLOv8 uses COCO dataset by default)
        self.class_names = self.model.names
        
        # Generate colors for each class
        self.colors = self._generate_colors(len(self.class_names))
        
        print(f"✅ YOLOv8 detector initialized with {len(self.class_names)} classes")
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate random colors for different classes."""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(num_classes):
            color = np.random.randint(0, 255, size=3)
            colors.append(tuple(map(int, color)))
        return colors
    
    def detect_objects(self, frame: np.ndarray) -> Tuple[List, List, List, List]:
        """
        Detect objects in the given frame using YOLOv8.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (boxes, confidences, class_ids, class_names)
        """
        try:
            # Run YOLOv8 inference
            results = self.model(frame, 
                               imgsz=self.imgsz,
                               conf=self.confidence_threshold,
                               device=self.device,
                               half=self.half,
                               verbose=False)
            
            boxes = []
            confidences = []
            class_ids = []
            class_names = []
            
            # Process results
            if results and len(results) > 0:
                result = results[0]  # Get first result
                
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Convert to xywh format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Get confidence and class ID
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names[class_id]
                        
                        boxes.append([x, y, w, h])
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        class_names.append(class_name)
            
            return boxes, confidences, class_ids, class_names
            
        except Exception as e:
            print(f"Error during detection: {e}")
            return [], [], [], []
    
    def draw_detections(self, frame: np.ndarray, boxes: List, confidences: List, 
                       class_ids: List, class_names: List) -> np.ndarray:
        """
        Draw detection boxes and labels on the frame.
        
        Args:
            frame: Input frame
            boxes: Detection boxes in [x, y, w, h] format
            confidences: Detection confidences
            class_ids: Detected class IDs
            class_names: Detected class names
            
        Returns:
            Frame with drawn detections
        """
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            confidence = confidences[i]
            class_id = class_ids[i]
            class_name = class_names[i]
            
            # Get color for this class
            color = self.colors[class_id] if class_id < len(self.colors) else (255, 255, 255)
            
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Create label text
            if config.SHOW_CONFIDENCE:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Draw label background
            cv2.rectangle(frame, (x, y - text_height - 10), 
                         (x + text_width, y), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame

class CameraApp:
    """Main camera application for real-time object detection with YOLOv8."""
    
    def __init__(self, camera_index: int = 0, model_name: str = "yolov8n.pt"):
        """
        Initialize the camera application.
        
        Args:
            camera_index: Index of the camera to use (0 for default)
            model_name: YOLOv8 model to use
        """
        self.camera_index = camera_index
        self.cap = None
        
        # Initialize YOLOv8 detector
        try:
            self.detector = YOLOv8Detector(
                model_name=model_name,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                device=config.YOLO_DEVICE,
                imgsz=config.YOLO_IMGSZ,
                half=config.YOLO_HALF,
                verbose=config.YOLO_VERBOSE
            )
        except Exception as e:
            print(f"❌ Failed to initialize YOLOv8 detector: {e}")
            print("Please install ultralytics: pip install ultralytics")
            raise

        # FPS calculation
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Create screenshot directory
        os.makedirs(config.SCREENSHOT_PATH, exist_ok=True)
    
    def initialize_camera(self) -> bool:
        """
        Initialize the camera capture.

        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.camera_index)

            if not self.cap.isOpened():
                print(f"❌ Error: Could not open camera {self.camera_index}")
                return False

            # Set camera resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            
            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"✅ Camera {self.camera_index} initialized successfully")
            print(f"Resolution: {actual_width}x{actual_height}")
            return True

        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False

    def calculate_fps(self) -> float:
        """Calculate and return current FPS."""
        self.fps_counter += 1
        if self.fps_counter >= 30:  # Update FPS every 30 frames
            elapsed_time = time.time() - self.fps_start_time
            self.current_fps = self.fps_counter / elapsed_time
            self.fps_counter = 0
            self.fps_start_time = time.time()

        return self.current_fps

    def save_screenshot(self, frame: np.ndarray) -> str:
        """Save a screenshot of the current frame."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{config.SCREENSHOT_PATH}screenshot_{timestamp}.{config.SCREENSHOT_FORMAT}"

        try:
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving screenshot: {e}")
            return ""

    def run(self):
        """Run the main camera loop with YOLOv8 object detection."""
        if not self.initialize_camera():
            return

        print("\n🎥 Starting YOLOv8 object detection...")
        print("Controls:")
        print("  'q' or 'ESC' - Quit")
        print("  'space' - Take screenshot")
        print("  'r' - Reset FPS counter")

        try:
            while True:
                # Ensure camera is initialized
                if self.cap is None or not self.cap.isOpened():
                    print("❌ Error: Camera is not initialized or has been released.")
                    break
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("❌ Error: Could not read frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Detect objects using YOLOv8
                start_time = time.time()
                boxes, confidences, class_ids, class_names = self.detector.detect_objects(frame)
                detection_time = time.time() - start_time
                
                # Draw detections
                frame = self.detector.draw_detections(frame, boxes, confidences, class_ids, class_names)
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                if config.SHOW_FPS and fps > 0:
                    fps_text = f"FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms"
                    cv2.putText(frame, fps_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display detection count
                detection_count = len(boxes)
                if detection_count > 0:
                    count_text = f"Objects detected: {detection_count}"
                    cv2.putText(frame, count_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Add instructions
                instructions = "Controls: 'q'=quit | 'space'=screenshot | 'r'=reset FPS"
                cv2.putText(frame, instructions, (10, frame.shape[0] - 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('YOLOv8 Object Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # Space for screenshot
                    self.save_screenshot(frame)
                elif key == ord('r'):  # Reset FPS counter
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                    self.current_fps = 0
                    print("🔄 FPS counter reset")
                
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            # Clean up
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("✅ Camera released and windows closed")

def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(description='Real-time Camera Object Detection with YOLOv8')
    parser.add_argument('--camera', type=int, default=config.CAMERA_INDEX,
                       help=f'Camera index to use (default: {config.CAMERA_INDEX})')
    parser.add_argument('--model', type=str, default=config.YOLO_MODEL,
                       help=f'YOLOv8 model to use (default: {config.YOLO_MODEL})')
    parser.add_argument('--confidence', type=float, default=config.CONFIDENCE_THRESHOLD,
                       help=f'Confidence threshold for detection (default: {config.CONFIDENCE_THRESHOLD})')
    parser.add_argument('--device', type=str, default=config.YOLO_DEVICE,
                       help=f'Device to run inference on (default: {config.YOLO_DEVICE})')
    parser.add_argument('--imgsz', type=int, default=config.YOLO_IMGSZ,
                       help=f'Input image size (default: {config.YOLO_IMGSZ})')
    parser.add_argument('--half', action='store_true',
                       help='Use half precision (FP16) for faster inference')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed model loading info')
    
    args = parser.parse_args()
    
    # Override config with command line arguments
    config.CAMERA_INDEX = args.camera
    config.CONFIDENCE_THRESHOLD = args.confidence
    config.YOLO_DEVICE = args.device
    config.YOLO_IMGSZ = args.imgsz
    config.YOLO_HALF = args.half
    config.YOLO_VERBOSE = args.verbose
    
    print(f"🎥 YOLOv8 Object Detection")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.confidence}")
    print(f"Image size: {args.imgsz}")
    
    # Check if YOLO is available
    if not YOLO_AVAILABLE:
        print("\n❌ Ultralytics YOLO is not installed!")
        print("Install it with: pip install ultralytics")
        return
    
    # Create and run the camera application
    try:
        app = CameraApp(camera_index=args.camera, model_name=args.model)
        app.run()
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is not being used by another application")
        print("2. Try a different camera index: python main.py --camera 1")
        print("3. Check if ultralytics is installed: pip install ultralytics")

if __name__ == "__main__":
    main()



