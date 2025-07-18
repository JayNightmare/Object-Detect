#!/usr/bin/env python3
"""
Demo script for object tracking functionality

This script demonstrates how to use the object tracking system
that remembers important objects and their locations.
"""

import time
from object_tracker import ObjectTracker, TrackedObject
import config


def demo_object_tracking():
    """Demonstrate object tracking functionality."""
    print("🎯 Object Tracking Demo")
    print("=" * 50)

    # Initialize tracker
    tracker = ObjectTracker(
        important_objects=config.IMPORTANT_OBJECTS[:5],  # Use first 5 objects for demo
        memory_duration=60,  # 1 minute for demo
        min_confidence=0.5,
        distance_threshold=50,
        history_file="demo_tracking_history.json",
        enable_logging=True,
        max_tracked_objects=100,
    )

    # Set demo frame dimensions
    tracker.set_frame_dimensions(640, 480)

    print(f"Tracking {len(tracker.important_objects)} object types:")
    for obj_type in sorted(tracker.important_objects):
        print(f"  • {obj_type}")

    # Simulate some detections
    print("\n📍 Simulating object detections...")

    # Simulate person detection in center
    boxes = [[200, 150, 100, 200]]  # x, y, w, h
    confidences = [0.85]
    class_ids = [0]
    class_names = ["person"]

    print("Detection 1: Person in center zone")
    tracked = tracker.update_tracking(boxes, confidences, class_ids, class_names)

    for obj_id, obj in tracked.items():
        print(
            f"  Tracked: {obj.class_name} at {obj.zone} (confidence: {obj.confidence:.2f})"
        )

    # Wait a bit
    time.sleep(2)

    # Simulate person moving to different location
    boxes = [[350, 100, 100, 200]]  # Moved to top-right
    confidences = [0.80]
    class_ids = [0]
    class_names = ["person"]

    print("\nDetection 2: Person moved to top-right zone")
    tracked = tracker.update_tracking(boxes, confidences, class_ids, class_names)

    for obj_id, obj in tracked.items():
        print(
            f"  Tracked: {obj.class_name} at {obj.zone} (confidence: {obj.confidence:.2f})"
        )
        print(f"  Times detected: {obj.times_detected}")

    # Wait a bit
    time.sleep(2)

    # Simulate laptop detection
    boxes = [[50, 300, 150, 100]]  # Laptop in bottom-left
    confidences = [0.75]
    class_ids = [63]  # Laptop class ID in COCO
    class_names = ["laptop"]

    print("\nDetection 3: Laptop detected in bottom-left zone")
    tracked = tracker.update_tracking(boxes, confidences, class_ids, class_names)

    for obj_id, obj in tracked.items():
        print(
            f"  Tracked: {obj.class_name} at {obj.zone} (confidence: {obj.confidence:.2f})"
        )

    # Wait to simulate time passing
    print("\n⏰ Waiting 3 seconds (simulating time passing)...")
    time.sleep(3)

    # Show tracking statistics
    print("\n📊 Tracking Statistics:")
    stats = tracker.get_tracking_statistics()
    print(f"  Total active objects: {stats['total_active_objects']}")
    print(f"  Total tracked objects: {stats['total_tracked_objects']}")
    print(f"  Object counts by class: {stats['object_counts_by_class']}")
    print(f"  Memory usage: {stats['memory_usage_ratio']:.1%}")

    # Show last seen information
    print("\n🔍 Last Seen Information:")
    active_objects = tracker.get_active_objects()

    if not active_objects:
        print("  No objects currently being tracked.")
    else:
        for obj_id, obj in active_objects.items():
            time_ago = tracker.format_time_ago(obj.last_seen)
            print(f"  • {obj.class_name}: {obj.zone} ({time_ago})")

    # Save tracking history
    print("\n💾 Saving tracking history...")
    if tracker.save_history():
        print("  ✅ History saved successfully")
    else:
        print("  ❌ Failed to save history")

    print("\n🎯 Demo completed!")
    print("\nIn the real application, you would:")
    print("  • Press 't' to show tracking info")
    print("  • Press 's' to save tracking history")
    print("  • Press 'i' to show tracking statistics")
    print("  • See visual overlays showing where objects were last seen")


def demo_zone_detection():
    """Demonstrate zone detection functionality."""
    print("\n🗺️ Zone Detection Demo")
    print("=" * 30)

    tracker = ObjectTracker(important_objects=["person"], memory_duration=60)
    tracker.set_frame_dimensions(640, 480)

    # Test different positions
    test_positions = [
        (100, 100, "top-left"),
        (320, 100, "top-center"),
        (500, 100, "top-right"),
        (100, 240, "center-left"),
        (320, 240, "center"),
        (500, 240, "center-right"),
        (100, 380, "bottom-left"),
        (320, 380, "bottom-center"),
        (500, 380, "bottom-right"),
    ]

    for x, y, expected_zone in test_positions:
        detected_zone = tracker.get_zone(x, y)
        status = "✅" if detected_zone.value == expected_zone else "❌"
        print(
            f"  {status} Position ({x:3}, {y:3}) -> {detected_zone.value:12} (expected: {expected_zone})"
        )


if __name__ == "__main__":
    try:
        demo_object_tracking()
        demo_zone_detection()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
