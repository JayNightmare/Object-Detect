#!/usr/bin/env python3
"""
Camera Object Detection Application with YOLOv11

This application uses your computer's camera to detect objects in real-time
using YOLOv11 from Ultralytics.
"""

import cv2
import numpy as np
import argparse
import time
import os
import logging
from typing import List, Tuple, Optional
from pathlib import Path

# Import Scene Interpreter
try:
    from scene_interpreter import SceneInterpreter

    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("Warning: scene_interpreter module not available. AI features disabled.")

# Import object tracker
try:
    from object_tracker import ObjectTracker, ObjectTrackerError

    TRACKER_AVAILABLE = True
except ImportError:
    TRACKER_AVAILABLE = False
    print("Warning: object_tracker module not available. Tracking features disabled.")

# Import YOLO from Ultralytics (YOLOv11 support)
try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print(
        "Warning: Ultralytics YOLO not available. Install with: pip install ultralytics"
    )

# Import project config
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using default settings.")

    class config:
        CAMERA_INDEX = 0
        CAMERA_WIDTH = 640
        CAMERA_HEIGHT = 480
        CONFIDENCE_THRESHOLD = 0.5
        NMS_THRESHOLD = 0.4
        MODEL_TYPE = "yolov11"
        YOLO_MODEL = "yolo11n.pt"
        YOLO_DEVICE = "cpu"
        YOLO_IMGSZ = 640
        YOLO_HALF = False
        YOLO_VERBOSE = False
        SHOW_FPS = True
        SHOW_CONFIDENCE = True
        SCREENSHOT_PATH = "screenshots/"
        SCREENSHOT_FORMAT = "jpg"

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
        ]
        TRACKING_MEMORY_DURATION = 300  # seconds to remember object locations
        TRACKING_MIN_CONFIDENCE = 0.8  # minimum confidence to track an object
        TRACKING_DISTANCE_THRESHOLD = (
            100  # pixels - objects closer than this are considered same instance
        )
        TRACKING_MAX_OBJECTS = 1000  # maximum number of objects to track simultaneously
        SHOW_LAST_SEEN_INFO = True
        TRACKING_HISTORY_FILE = "object_tracking_history.json"
        TRACKING_ENABLE_LOGGING = True

        # Recording settings
        ENABLE_RECORDING = False
        OUTPUT_VIDEO_PATH = "output/"
        VIDEO_CODEC = "mp4v"
        VIDEO_FPS = 30.0
        VIDEO_FILENAME_PREFIX = "detection_output"


class YOLODetector:
    """Real-time object detection using YOLOv11 from Ultralytics."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        imgsz: int = 640,
        half: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize the YOLO detector.

        Args:
            model_name: YOLO model name (yolo11n.pt, yolo11s.pt, etc.)
            confidence_threshold: Minimum confidence for object detection
            device: Device to run inference on ('cpu', 'cuda', 'mps')
            imgsz: Input image size
            half: Use half precision (FP16) for faster inference
            verbose: Show detailed model loading info
        """
        if not YOLO_AVAILABLE:
            raise ImportError(
                "Ultralytics YOLO is not available. Install with: pip install ultralytics"
            )

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.imgsz = imgsz
        self.half = half
        self.verbose = verbose

        # Load YOLO model (v11 default)
        print(f"Loading YOLO model: {model_name}")
        try:
            self.model = YOLO(model_name)
            if verbose:
                print(f"✅ Model loaded successfully: {model_name}")
                print(f"Device: {device}")
                print(f"Image size: {imgsz}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

        # COCO class names (YOLO models use the COCO dataset by default)
        self.class_names = self.model.names

        # Generate colors for each class
        self.colors = self._generate_colors(len(self.class_names))

        print(f"✅ YOLO detector initialized with {len(self.class_names)} classes")

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
        Detect objects in the given frame using YOLOv11.

        Args:
            frame: Input frame from camera

        Returns:
            Tuple of (boxes, confidences, class_ids, class_names)
        """
        try:
            # Run YOLO inference
            results = self.model(
                frame,
                imgsz=self.imgsz,
                conf=self.confidence_threshold,
                device=self.device,
                half=self.half,
                verbose=False,
            )

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

    def draw_detections(
        self,
        frame: np.ndarray,
        boxes: List,
        confidences: List,
        class_ids: List,
        class_names: List,
    ) -> np.ndarray:
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
            color = (
                self.colors[class_id]
                if class_id < len(self.colors)
                else (255, 255, 255)
            )

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Create label text
            if config.SHOW_CONFIDENCE:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Draw label background
            cv2.rectangle(
                frame, (x, y - text_height - 10), (x + text_width, y), color, -1
            )

            # Draw label text
            cv2.putText(
                frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2
            )

        return frame


class CameraApp:
    """Main camera application for real-time object detection with YOLOv11."""

    def __init__(self, source: str = "0", model_name: str = "yolo11n.pt"):
        """
        Initialize the camera application.

        Args:
            source: Camera index (as string "0", "1") or video file path
            model_name: YOLO model to use
        """
        self.source = source
        # Try to convert to int if it's a digit (for camera index)
        if str(source).isdigit():
            self.source = int(source)

        self.cap = None
        self.video_writer = None

        # Initialize YOLO detector
        try:
            self.detector = YOLODetector(
                model_name=model_name,
                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                device=config.YOLO_DEVICE,
                imgsz=config.YOLO_IMGSZ,
                half=config.YOLO_HALF,
                verbose=config.YOLO_VERBOSE,
            )
        except Exception as e:
            print(f"❌ Failed to initialize YOLO detector: {e}")
            print("Please install ultralytics: pip install ultralytics")
            raise

        # Initialize object tracker (following Azure best practices)
        self.tracker = None
        if (
            TRACKER_AVAILABLE
            and hasattr(config, "TRACKING_ENABLED")
            and config.TRACKING_ENABLED
        ):
            try:
                self.tracker = ObjectTracker(
                    important_objects=config.IMPORTANT_OBJECTS,
                    memory_duration=config.TRACKING_MEMORY_DURATION,
                    min_confidence=config.TRACKING_MIN_CONFIDENCE,
                    distance_threshold=config.TRACKING_DISTANCE_THRESHOLD,
                    history_file=config.TRACKING_HISTORY_FILE,
                    enable_logging=config.TRACKING_ENABLE_LOGGING,
                    max_tracked_objects=config.TRACKING_MAX_OBJECTS,
                )
                print("✅ Object tracking initialized successfully")
            except ObjectTrackerError as e:
                print(f"⚠️ Warning: Object tracking initialization failed: {e}")
                print("Continuing without tracking features...")
                self.tracker = None
            except Exception as e:
                print(f"⚠️ Warning: Unexpected error initializing tracker: {e}")
                print("Continuing without tracking features...")
                self.tracker = None

        # Initialize AI Scene Interpreter
        self.scene_interpreter = None
        self.chat_mode = False
        self.user_input = ""
        self.last_thought_time = 0

        if AI_AVAILABLE and hasattr(config, "AI_ENABLED") and config.AI_ENABLED:
            try:
                self.scene_interpreter = SceneInterpreter(
                    api_key=config.OPENROUTER_API_KEY,
                    model=config.OPENROUTER_MODEL,
                    system_prompt=getattr(config, "AI_SYSTEM_PROMPT", None),
                )
                print("✅ AI Scene Interpreter initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: AI initialization failed: {e}")
                self.scene_interpreter = None

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
            self.cap = cv2.VideoCapture(self.source)

            if not self.cap.isOpened():
                print(f"❌ Error: Could not open source {self.source}")
                return False

            # Set camera resolution (only works for webcams, ignored for video files)
            if isinstance(self.source, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

            # Get actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            print(f"✅ Source {self.source} initialized successfully")
            print(f"Resolution: {actual_width}x{actual_height}")

            # Initialize video writer if recording is enabled
            self.initialize_writer(actual_width, actual_height)

            return True

        except Exception as e:
            print(f"❌ Error initializing camera: {e}")
            return False

    def initialize_writer(self, width: int, height: int):
        """Initialize video writer for recording."""
        if hasattr(config, "ENABLE_RECORDING") and config.ENABLE_RECORDING:
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                os.makedirs(config.OUTPUT_VIDEO_PATH, exist_ok=True)

                # Handle filename prefix
                prefix = getattr(config, "VIDEO_FILENAME_PREFIX", "detection_output")
                filename = (
                    f"{os.path.join(config.OUTPUT_VIDEO_PATH, prefix)}_{timestamp}.mp4"
                )

                # Get codec
                codec_str = getattr(config, "VIDEO_CODEC", "mp4v")
                fourcc = cv2.VideoWriter_fourcc(*codec_str)

                fps = getattr(config, "VIDEO_FPS", 30.0)

                self.video_writer = cv2.VideoWriter(
                    filename, fourcc, fps, (width, height)
                )
                print(f"🎥 Recording enabled: {filename}")
            except Exception as e:
                print(f"❌ Error initializing video writer: {e}")
                self.video_writer = None

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
        filename = (
            f"{config.SCREENSHOT_PATH}screenshot_{timestamp}.{config.SCREENSHOT_FORMAT}"
        )

        try:
            cv2.imwrite(filename, frame)
            print(f"📸 Screenshot saved: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Error saving screenshot: {e}")
            return ""

    def draw_tracking_info(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw tracking information on the frame following Azure best practices.

        Args:
            frame: Input frame to draw on

        Returns:
            Frame with tracking information overlaid
        """
        if not self.tracker:
            return frame

        try:
            active_objects = self.tracker.get_active_objects()
            y_offset = 100

            for obj_id, tracked_obj in active_objects.items():
                current_time = time.time()
                time_ago = self.tracker.format_time_ago(tracked_obj.last_seen)

                # Determine if object is currently visible (seen within last 2 seconds)
                is_current = (current_time - tracked_obj.last_seen) < 2

                if is_current:
                    status_text = (
                        f"🟢 {tracked_obj.class_name} - {tracked_obj.zone} (now)"
                    )
                    color = (0, 255, 0)  # Green for current
                else:
                    status_text = (
                        f"🔴 {tracked_obj.class_name} - {tracked_obj.zone} ({time_ago})"
                    )
                    color = (0, 100, 255)  # Orange for last seen

                cv2.putText(
                    frame,
                    status_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                y_offset += 20

                # Prevent overlay from going off-screen
                if y_offset > frame.shape[0] - 50:
                    break

            return frame

        except Exception as e:
            print(f"⚠️ Warning: Failed to draw tracking info: {e}")
            return frame

    def draw_ai_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw AI response and chat interface.
        """
        if not self.scene_interpreter:
            return frame

        h, w = frame.shape[:2]

        # Draw latest response
        response = self.scene_interpreter.get_latest_response()
        if response:
            # Split response into lines to fit screen
            max_width = 40  # characters (reduced for right side)
            lines = []
            for line in response.split("\n"):
                while len(line) > max_width:
                    lines.append(line[:max_width])
                    line = line[max_width:]
            lines.append(line)

            # Calculate dimensions for right-side overlay
            text_height = len(lines) * 25
            overlay_width = 400  # Fixed width for right side overlay
            x_start = w - overlay_width

            # Draw background for response on the right side
            cv2.rectangle(frame, (x_start, 75), (w, text_height + 90), (0, 0, 0), -1)

            for i, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (x_start + 10, 100 + 25 * i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # Draw chat input if in chat mode
        if self.chat_mode:
            # Input box at bottom
            cv2.rectangle(frame, (0, h - 60), (w, h), (50, 50, 50), -1)
            cv2.putText(
                frame,
                f"User: {self.user_input}_",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Instructions
            cv2.putText(
                frame,
                "Type your message. ENTER to send, ESC to cancel.",
                (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
            )

        return frame

    def handle_tracking_commands(self, key: int) -> None:
        """
        Handle keyboard commands for tracking features.

        Args:
            key: Pressed key code
        """
        if not self.tracker:
            return

        try:
            if key == ord("t"):  # Show tracking info
                print("\n📍 Currently Tracked Objects:")
                active_objects = self.tracker.get_active_objects()

                if not active_objects:
                    print("No objects currently being tracked.")
                else:
                    for obj_id, tracked_obj in active_objects.items():
                        time_ago = self.tracker.format_time_ago(tracked_obj.last_seen)
                        print(
                            f"  • {tracked_obj.class_name}: {tracked_obj.zone} ({time_ago})"
                        )
                        print(
                            f"    Detected {tracked_obj.times_detected} times, confidence: {tracked_obj.confidence:.2f}"
                        )

            elif key == ord("s"):  # Save tracking history
                if self.tracker.save_history():
                    print("💾 Tracking history saved successfully")
                else:
                    print("❌ Failed to save tracking history")

            elif key == ord("i"):  # Show tracking statistics
                stats = self.tracker.get_tracking_statistics()
                print("\n📊 Tracking Statistics:")
                print(f"  • Total active objects: {stats['total_active_objects']}")
                print(f"  • Total tracked objects: {stats['total_tracked_objects']}")
                print(f"  • Memory usage: {stats['memory_usage_ratio']:.1%}")
                print(f"  • Object counts by class: {stats['object_counts_by_class']}")

        except Exception as e:
            print(f"⚠️ Warning: Failed to handle tracking command: {e}")

    def find_object_command(self) -> None:
        """
        Handle object finding functionality.
        Note: This is a simplified version. In a production system,
        you might want to implement this using a separate thread or GUI.
        """
        if not self.tracker:
            print("⚠️ Object tracking is not available")
            return

        try:
            print("\nAvailable object types to search for:")
            for obj_type in sorted(config.IMPORTANT_OBJECTS):
                print(f"  • {obj_type}")

            print(
                "\nPress 'f' + object initial to find (e.g., 'fp' for person, 'fl' for laptop)"
            )

        except Exception as e:
            print(f"⚠️ Warning: Failed to show find object menu: {e}")

    def run(self):
        """Run the main camera loop with YOLOv11 object detection."""
        if not self.initialize_camera():
            return

        # Set frame dimensions for tracker
        if self.tracker:
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.tracker.set_frame_dimensions(actual_width, actual_height)

        print("\n🎥 Starting YOLOv11 object detection...")
        print("Controls:")
        print("  'q' or 'ESC' - Quit")
        print("  'space' - Take screenshot")
        print("  'r' - Reset FPS counter")
        if self.tracker:
            print("  't' - Show tracking info")
            print("  's' - Save tracking history")
            print("  'i' - Show tracking statistics")

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

                # Detect objects using YOLOv11
                start_time = time.time()
                boxes, confidences, class_ids, class_names = (
                    self.detector.detect_objects(frame)
                )
                detection_time = time.time() - start_time

                # Update object tracking
                if self.tracker:
                    try:
                        tracked_objects = self.tracker.update_tracking(
                            boxes, confidences, class_ids, class_names, frame=frame
                        )

                        # AI Scene Interpretation
                        if self.scene_interpreter:
                            current_time = time.time()

                            # Passive mode: periodic "thinking"
                            if not self.chat_mode:
                                interval = getattr(config, "AI_THOUGHT_INTERVAL", 10.0)
                                if current_time - self.last_thought_time > interval:
                                    self.scene_interpreter.interpret_scene(
                                        tracked_objects, mode="passive"
                                    )
                                    self.last_thought_time = current_time

                    except Exception as e:
                        print(f"⚠️ Warning: Tracking/AI update failed: {e}")

                # Draw detections
                frame = self.detector.draw_detections(
                    frame, boxes, confidences, class_ids, class_names
                )

                # Draw tracking information
                if (
                    self.tracker
                    and hasattr(config, "SHOW_LAST_SEEN_INFO")
                    and config.SHOW_LAST_SEEN_INFO
                ):
                    frame = self.draw_tracking_info(frame)

                # Draw AI Overlay
                if self.scene_interpreter:
                    frame = self.draw_ai_overlay(frame)

                # Calculate and display FPS
                fps = self.calculate_fps()
                if config.SHOW_FPS and fps > 0:
                    fps_text = (
                        f"FPS: {fps:.1f} | Detection: {detection_time*1000:.1f}ms"
                    )
                    cv2.putText(
                        frame,
                        fps_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Display detection count
                detection_count = len(boxes)
                if detection_count > 0:
                    count_text = f"Objects detected: {detection_count}"
                    cv2.putText(
                        frame,
                        count_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                # Add instructions
                if self.tracker:
                    instructions = "Controls: 'q'=quit | 'space'=screenshot | 'r'=reset FPS | 't'=tracking | 'i'=stats"
                    if self.scene_interpreter:
                        instructions += " | 'c'=chat"
                else:
                    instructions = (
                        "Controls: 'q'=quit | 'space'=screenshot | 'r'=reset FPS"
                    )
                cv2.putText(
                    frame,
                    instructions,
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                )

                # Display frame
                cv2.imshow("Object Detection w/ Yolo", frame)

                # Write frame to video
                if self.video_writer:
                    self.video_writer.write(frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                # Chat mode input handling
                if self.chat_mode:
                    if key == 27:  # ESC to exit chat mode
                        self.chat_mode = False
                        self.user_input = ""
                    elif key == 13:  # ENTER to send
                        if self.user_input.strip():
                            # Get current tracking data for context
                            active_objects = (
                                self.tracker.get_active_objects()
                                if self.tracker
                                else {}
                            )
                            self.scene_interpreter.interpret_scene(
                                active_objects, user_prompt=self.user_input, mode="chat"
                            )
                            self.user_input = ""
                    elif key == 8:  # Backspace
                        self.user_input = self.user_input[:-1]
                    elif 32 <= key <= 126:  # Printable characters
                        self.user_input += chr(key)
                else:
                    # Normal controls
                    if key == ord("q") or key == 27:  # 'q' or ESC
                        break
                    elif key == ord(" "):  # Space for screenshot
                        self.save_screenshot(frame)
                    elif key == ord("r"):  # Reset FPS counter
                        self.fps_counter = 0
                        self.fps_start_time = time.time()
                        self.current_fps = 0
                        print("🔄 FPS counter reset")
                    elif key == ord("c") and self.scene_interpreter:  # Toggle chat mode
                        self.chat_mode = True
                        self.user_input = ""
                    elif self.tracker:
                        self.handle_tracking_commands(key)

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            # Save tracking history before exit
            if self.tracker:
                try:
                    self.tracker.save_history()
                    print("💾 Tracking history saved on exit")
                except Exception as e:
                    print(f"⚠️ Warning: Failed to save tracking history: {e}")

            # Clean up
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            print("✅ Camera released and windows closed")


def main():
    """Main function to parse arguments and run the application."""
    parser = argparse.ArgumentParser(
        description="Real-time Camera Object Detection with YOLOv11"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(config.CAMERA_INDEX),
        help=f"Camera index (e.g., '0') or video file path (default: {config.CAMERA_INDEX})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.YOLO_MODEL,
        help=f"YOLO model to use (default: {config.YOLO_MODEL})",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for detection (default: {config.CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.YOLO_DEVICE,
        help=f"Device to run inference on (default: {config.YOLO_DEVICE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=config.YOLO_IMGSZ,
        help=f"Input image size (default: {config.YOLO_IMGSZ})",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use half precision (FP16) for faster inference",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed model loading info"
    )

    args = parser.parse_args()

    # Override config with command line arguments
    # config.CAMERA_INDEX is kept for backward compatibility but source is used
    if args.source.isdigit():
        config.CAMERA_INDEX = int(args.source)

    config.CONFIDENCE_THRESHOLD = args.confidence
    config.YOLO_DEVICE = args.device
    config.YOLO_IMGSZ = args.imgsz
    config.YOLO_HALF = args.half
    config.YOLO_VERBOSE = args.verbose

    # Print final configuration
    print("\n📸 Final Configuration:")
    print(f"  Camera Source: {config.CAMERA_INDEX}")
    print(f"  Model: {config.YOLO_MODEL}")
    print(f"  Confidence Threshold: {config.CONFIDENCE_THRESHOLD}")
    print(f"  Device: {config.YOLO_DEVICE}")
    print(f"  Image Size: {config.YOLO_IMGSZ}")
    print(f"  Half Precision: {'Enabled' if config.YOLO_HALF else 'Disabled'}")
    print(f"  Verbose Mode: {'Enabled' if config.YOLO_VERBOSE else 'Disabled'}")

    # Check YOLO availability
    if not YOLO_AVAILABLE:
        print("\n❌ Ultralytics YOLO is not installed!")
        print("Install it with: pip install ultralytics")
        return

    # Create and run the camera application
    try:
        app = CameraApp(source=args.source, model_name=args.model)
        app.run()
    except Exception as e:
        print(f"❌ Error running application: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure your camera is not being used by another application")
        print("2. Try a different camera index/file: python main.py --source 1")
        print("3. Check if ultralytics is installed: pip install ultralytics")


if __name__ == "__main__":
    main()
