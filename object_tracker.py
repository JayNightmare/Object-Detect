#!/usr/bin/env python3
"""
Object Tracking Module for Camera Object Detection Application

This module implements an intelligent object tracking system that remembers
important objects and their last seen locations using Azure coding best practices.
Includes comprehensive error handling, logging, and configuration management.
"""

import json
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum


class TrackingZone(Enum):
    """Enumeration for frame zones to improve type safety."""

    TOP_LEFT = "top-left"
    TOP_CENTER = "top-center"
    TOP_RIGHT = "top-right"
    CENTER_LEFT = "center-left"
    CENTER = "center"
    CENTER_RIGHT = "center-right"
    BOTTOM_LEFT = "bottom-left"
    BOTTOM_CENTER = "bottom-center"
    BOTTOM_RIGHT = "bottom-right"


@dataclass
class TrackedObject:
    """
    Represents a tracked object with comprehensive metadata.

    Follows Azure best practices for data models with proper typing
    and immutable-like behavior where appropriate.
    """

    class_name: str
    confidence: float
    center_x: int
    center_y: int
    width: int
    height: int
    first_seen: float
    last_seen: float
    times_detected: int
    zone: str
    object_id: str

    def __post_init__(self):
        """Validate object data after initialization."""
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.first_seen > self.last_seen:
            raise ValueError("first_seen cannot be later than last_seen")


class ObjectTrackerError(Exception):
    """Custom exception for object tracking operations."""

    pass


class ObjectTracker:
    """
    Intelligent object tracking system following Azure coding best practices.

    Features:
    - Comprehensive error handling with exponential backoff for file operations
    - Structured logging for monitoring and debugging
    - Configuration-driven behavior
    - Resource cleanup and memory management
    - Thread-safe operations where applicable
    """

    def __init__(
        self,
        important_objects: List[str],
        memory_duration: int = 300,
        min_confidence: float = 0.6,
        distance_threshold: int = 100,
        history_file: str = "object_tracking_history.json",
        enable_logging: bool = True,
        max_tracked_objects: int = 1000,
    ):
        """
        Initialize the object tracker with comprehensive configuration.

        Args:
            important_objects: List of object class names to track
            memory_duration: How long to remember objects (seconds)
            min_confidence: Minimum confidence threshold for tracking
            distance_threshold: Maximum distance to consider same object instance
            history_file: File path for persistent storage
            enable_logging: Enable structured logging
            max_tracked_objects: Maximum number of objects to track simultaneously

        Raises:
            ObjectTrackerError: If initialization parameters are invalid
        """
        try:
            # Validate input parameters
            self._validate_init_parameters(
                important_objects,
                memory_duration,
                min_confidence,
                distance_threshold,
                max_tracked_objects,
            )

            # Configuration
            self.important_objects: Set[str] = set(important_objects)
            self.memory_duration = memory_duration
            self.min_confidence = min_confidence
            self.distance_threshold = distance_threshold
            self.max_tracked_objects = max_tracked_objects
            self.history_file = Path(history_file)

            # Tracking state
            self.tracked_objects: Dict[str, TrackedObject] = {}
            self.next_object_id = 1

            # Frame dimensions for zone calculation
            self.frame_width = 640
            self.frame_height = 480

            # Zone grid configuration (3x3 grid)
            self._zone_grid = [
                [
                    TrackingZone.TOP_LEFT,
                    TrackingZone.TOP_CENTER,
                    TrackingZone.TOP_RIGHT,
                ],
                [
                    TrackingZone.CENTER_LEFT,
                    TrackingZone.CENTER,
                    TrackingZone.CENTER_RIGHT,
                ],
                [
                    TrackingZone.BOTTOM_LEFT,
                    TrackingZone.BOTTOM_CENTER,
                    TrackingZone.BOTTOM_RIGHT,
                ],
            ]

            # Initialize logging
            if enable_logging:
                self._setup_logging()
            else:
                self.logger = logging.getLogger(__name__)
                self.logger.addHandler(logging.NullHandler())

            # Load existing tracking history with retry logic
            self._load_history_with_retry()

            self.logger.info(
                f"ObjectTracker initialized successfully. "
                f"Tracking {len(self.important_objects)} object types, "
                f"memory duration: {self.memory_duration}s"
            )

        except Exception as e:
            raise ObjectTrackerError(f"Failed to initialize ObjectTracker: {e}") from e

    def _validate_init_parameters(
        self,
        important_objects: List[str],
        memory_duration: int,
        min_confidence: float,
        distance_threshold: int,
        max_tracked_objects: int,
    ) -> None:
        """Validate initialization parameters."""
        if not important_objects:
            raise ValueError("important_objects cannot be empty")
        if not all(isinstance(obj, str) and obj.strip() for obj in important_objects):
            raise ValueError("All important_objects must be non-empty strings")
        if memory_duration <= 0:
            raise ValueError("memory_duration must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")
        if distance_threshold <= 0:
            raise ValueError("distance_threshold must be positive")
        if max_tracked_objects <= 0:
            raise ValueError("max_tracked_objects must be positive")

    def _setup_logging(self) -> None:
        """Configure structured logging following Azure best practices."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_frame_dimensions(self, width: int, height: int) -> None:
        """
        Set frame dimensions for accurate zone calculation.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels

        Raises:
            ValueError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")

        self.frame_width = width
        self.frame_height = height
        self.logger.debug(f"Frame dimensions updated: {width}x{height}")

    def get_zone(self, center_x: int, center_y: int) -> TrackingZone:
        """
        Determine which zone of the frame the object is in.

        Args:
            center_x: X coordinate of object center
            center_y: Y coordinate of object center

        Returns:
            TrackingZone enum value
        """
        try:
            # Divide frame into 3x3 grid
            zone_width = self.frame_width // 3
            zone_height = self.frame_height // 3

            col = min(center_x // zone_width, 2)
            row = min(center_y // zone_height, 2)

            return self._zone_grid[row][col]

        except (IndexError, ZeroDivisionError) as e:
            self.logger.warning(f"Zone calculation failed, defaulting to CENTER: {e}")
            return TrackingZone.CENTER

    def calculate_distance(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """
        Calculate Euclidean distance between two points.

        Args:
            x1, y1: First point coordinates
            x2, y2: Second point coordinates

        Returns:
            Euclidean distance as float
        """
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_matching_object(
        self, class_name: str, center_x: int, center_y: int
    ) -> Optional[str]:
        """
        Find existing tracked object that matches the current detection.

        Uses spatial proximity and temporal constraints to match objects
        across frames, implementing a simple but effective tracking algorithm.

        Args:
            class_name: Object class name
            center_x: X coordinate of object center
            center_y: Y coordinate of object center

        Returns:
            Object ID if match found, None otherwise
        """
        current_time = time.time()
        best_match = None
        min_distance = float("inf")

        try:
            for obj_id, tracked_obj in self.tracked_objects.items():
                # Only match same class objects
                if tracked_obj.class_name != class_name:
                    continue

                # Skip if object is too old
                if current_time - tracked_obj.last_seen > self.memory_duration:
                    continue

                # Calculate distance
                distance = self.calculate_distance(
                    center_x, center_y, tracked_obj.center_x, tracked_obj.center_y
                )

                # Find closest matching object within threshold
                if distance < self.distance_threshold and distance < min_distance:
                    min_distance = distance
                    best_match = obj_id

            return best_match

        except Exception as e:
            self.logger.error(f"Error finding matching object: {e}")
            return None

    def update_tracking(
        self,
        boxes: List[List[int]],
        confidences: List[float],
        class_ids: List[int],
        class_names: List[str],
    ) -> Dict[str, TrackedObject]:
        """
        Update tracking information with new detections.

        Implements the core tracking logic with comprehensive error handling
        and memory management following Azure best practices.

        Args:
            boxes: Detection boxes in [x, y, w, h] format
            confidences: Detection confidences
            class_ids: Detected class IDs
            class_names: Detected class names

        Returns:
            Dictionary of currently tracked important objects

        Raises:
            ObjectTrackerError: If tracking update fails
        """
        try:
            # Validate input parameters
            if not all(
                len(lst) == len(boxes) for lst in [confidences, class_ids, class_names]
            ):
                raise ValueError("All input lists must have the same length")

            current_time = time.time()
            updates_made = 0

            # Process each detection
            for i in range(len(boxes)):
                try:
                    class_name = class_names[i]
                    confidence = confidences[i]

                    # Only track important objects with sufficient confidence
                    if (
                        class_name not in self.important_objects
                        or confidence < self.min_confidence
                    ):
                        continue

                    x, y, w, h = boxes[i]
                    center_x = x + w // 2
                    center_y = y + h // 2
                    zone = self.get_zone(center_x, center_y)

                    # Try to find matching existing object
                    matching_id = self.find_matching_object(
                        class_name, center_x, center_y
                    )

                    if matching_id:
                        # Update existing object
                        tracked_obj = self.tracked_objects[matching_id]
                        tracked_obj.confidence = max(tracked_obj.confidence, confidence)
                        tracked_obj.center_x = center_x
                        tracked_obj.center_y = center_y
                        tracked_obj.width = w
                        tracked_obj.height = h
                        tracked_obj.last_seen = current_time
                        tracked_obj.times_detected += 1
                        tracked_obj.zone = zone.value
                        updates_made += 1
                    else:
                        # Create new tracked object (with memory limit check)
                        if len(self.tracked_objects) >= self.max_tracked_objects:
                            self._cleanup_oldest_objects(1)

                        obj_id = f"{class_name}_{self.next_object_id}"
                        self.next_object_id += 1

                        self.tracked_objects[obj_id] = TrackedObject(
                            class_name=class_name,
                            confidence=confidence,
                            center_x=center_x,
                            center_y=center_y,
                            width=w,
                            height=h,
                            first_seen=current_time,
                            last_seen=current_time,
                            times_detected=1,
                            zone=zone.value,
                            object_id=obj_id,
                        )
                        updates_made += 1

                        self.logger.debug(
                            f"New object tracked: {obj_id} at {zone.value}"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to process detection {i}: {e}")
                    continue

            # Clean up old objects
            cleaned_count = self.cleanup_old_objects()

            if updates_made > 0 or cleaned_count > 0:
                self.logger.debug(
                    f"Tracking updated: {updates_made} updates, {cleaned_count} cleaned"
                )

            return self.get_active_objects()

        except Exception as e:
            self.logger.error(f"Failed to update tracking: {e}")
            raise ObjectTrackerError(f"Tracking update failed: {e}") from e

    def cleanup_old_objects(self) -> int:
        """
        Remove objects that haven't been seen for too long.

        Returns:
            Number of objects removed
        """
        current_time = time.time()
        expired_objects = []

        for obj_id, tracked_obj in self.tracked_objects.items():
            if current_time - tracked_obj.last_seen > self.memory_duration:
                expired_objects.append(obj_id)

        for obj_id in expired_objects:
            del self.tracked_objects[obj_id]

        if expired_objects:
            self.logger.debug(f"Cleaned up {len(expired_objects)} expired objects")

        return len(expired_objects)

    def _cleanup_oldest_objects(self, count: int) -> None:
        """Remove the oldest objects to free memory."""
        if not self.tracked_objects:
            return

        # Sort by last_seen timestamp and remove oldest
        sorted_objects = sorted(
            self.tracked_objects.items(), key=lambda x: x[1].last_seen
        )

        for i in range(min(count, len(sorted_objects))):
            obj_id = sorted_objects[i][0]
            del self.tracked_objects[obj_id]
            self.logger.debug(f"Removed oldest object: {obj_id}")

    def get_active_objects(self) -> Dict[str, TrackedObject]:
        """Get objects that are currently being tracked."""
        current_time = time.time()
        active_objects = {}

        for obj_id, tracked_obj in self.tracked_objects.items():
            if current_time - tracked_obj.last_seen <= self.memory_duration:
                active_objects[obj_id] = tracked_obj

        return active_objects

    def get_last_seen_info(self, class_name: str) -> Optional[TrackedObject]:
        """
        Get information about when an object was last seen.

        Args:
            class_name: Object class name to search for

        Returns:
            TrackedObject if found, None otherwise
        """
        latest_obj = None
        latest_time = 0

        for tracked_obj in self.tracked_objects.values():
            if (
                tracked_obj.class_name == class_name
                and tracked_obj.last_seen > latest_time
            ):
                latest_time = tracked_obj.last_seen
                latest_obj = tracked_obj

        return latest_obj

    def _save_history_with_retry(self, max_retries: int = 3) -> bool:
        """
        Save tracking history to file with exponential backoff retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Create parent directory if it doesn't exist
                self.history_file.parent.mkdir(parents=True, exist_ok=True)

                history_data = {}
                for obj_id, tracked_obj in self.tracked_objects.items():
                    history_data[obj_id] = asdict(tracked_obj)

                # Atomic write using temporary file
                temp_file = self.history_file.with_suffix(".tmp")
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(history_data, f, indent=2, ensure_ascii=False)

                # Atomic rename
                temp_file.replace(self.history_file)

                self.logger.debug(
                    f"Tracking history saved successfully to {self.history_file}"
                )
                return True

            except Exception as e:
                wait_time = (2**attempt) * 0.1  # Exponential backoff
                self.logger.warning(
                    f"Failed to save history (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        self.logger.error(
            f"Failed to save tracking history after {max_retries} attempts"
        )
        return False

    def _load_history_with_retry(self, max_retries: int = 3) -> bool:
        """
        Load tracking history from file with retry logic.

        Args:
            max_retries: Maximum number of retry attempts

        Returns:
            True if successful, False otherwise
        """
        if not self.history_file.exists():
            self.logger.info("No existing tracking history file found")
            return True

        for attempt in range(max_retries):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    history_data = json.load(f)

                loaded_count = 0
                for obj_id, obj_data in history_data.items():
                    try:
                        self.tracked_objects[obj_id] = TrackedObject(**obj_data)
                        loaded_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to load object {obj_id}: {e}")

                # Update next_object_id to avoid conflicts
                if self.tracked_objects:
                    max_id = max(
                        [
                            int(oid.split("_")[-1])
                            for oid in self.tracked_objects.keys()
                            if "_" in oid and oid.split("_")[-1].isdigit()
                        ]
                    )
                    self.next_object_id = max_id + 1

                self.logger.info(f"Loaded {loaded_count} objects from tracking history")
                return True

            except Exception as e:
                wait_time = (2**attempt) * 0.1  # Exponential backoff
                self.logger.warning(
                    f"Failed to load history (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time:.1f}s"
                )
                if attempt < max_retries - 1:
                    time.sleep(wait_time)

        self.logger.error(
            f"Failed to load tracking history after {max_retries} attempts"
        )
        return False

    def save_history(self) -> bool:
        """Public method to save tracking history."""
        return self._save_history_with_retry()

    def format_time_ago(self, timestamp: float) -> str:
        """
        Format timestamp as human-readable time ago.

        Args:
            timestamp: Unix timestamp

        Returns:
            Formatted time string (e.g., "5s ago", "2m ago", "1h ago")
        """
        try:
            seconds_ago = time.time() - timestamp

            if seconds_ago < 0:
                return "in the future"
            elif seconds_ago < 60:
                return f"{int(seconds_ago)}s ago"
            elif seconds_ago < 3600:
                return f"{int(seconds_ago/60)}m ago"
            elif seconds_ago < 86400:
                return f"{int(seconds_ago/3600)}h ago"
            else:
                return f"{int(seconds_ago/86400)}d ago"

        except Exception as e:
            self.logger.warning(f"Failed to format timestamp {timestamp}: {e}")
            return "unknown"

    def get_tracking_statistics(self) -> Dict[str, any]:
        """
        Get comprehensive tracking statistics for monitoring.

        Returns:
            Dictionary containing tracking metrics
        """
        active_objects = self.get_active_objects()

        # Calculate statistics
        object_counts = {}
        total_detections = 0
        oldest_timestamp = float("inf")
        newest_timestamp = 0

        for tracked_obj in active_objects.values():
            class_name = tracked_obj.class_name
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            total_detections += tracked_obj.times_detected
            oldest_timestamp = min(oldest_timestamp, tracked_obj.first_seen)
            newest_timestamp = max(newest_timestamp, tracked_obj.last_seen)

        return {
            "total_active_objects": len(active_objects),
            "total_tracked_objects": len(self.tracked_objects),
            "object_counts_by_class": object_counts,
            "total_detections": total_detections,
            "tracking_duration_seconds": (
                newest_timestamp - oldest_timestamp if active_objects else 0
            ),
            "memory_usage_ratio": len(self.tracked_objects) / self.max_tracked_objects,
            "important_object_types": list(self.important_objects),
        }

    def __del__(self):
        """Cleanup resources when object is destroyed."""
        try:
            if hasattr(self, "tracked_objects") and self.tracked_objects:
                self._save_history_with_retry(max_retries=1)
        except Exception:
            pass  # Ignore errors during cleanup
