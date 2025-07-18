#!/usr/bin/env python3
"""
Demo script to test camera functionality without full object detection.

This script provides a simple camera test to verify that your camera is working
before running the full object detection application.
"""

import cv2
import sys


def test_camera(camera_index=0):
    """Test camera functionality."""
    print(f"Testing camera {camera_index}...")

    # Try to open the camera
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"❌ Error: Could not open camera {camera_index}")
        return False

    print(f"✅ Camera {camera_index} opened successfully!")
    print("Press 'q' to close the test window")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("❌ Error: Could not read frame from camera")
                break

            frame_count += 1

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Add some test text
            cv2.putText(
                frame,
                f"Camera Test - Frame {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Display the frame
            cv2.imshow("Camera Test", frame)

            # Check for quit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("✅ Test completed successfully!")
                break

    except KeyboardInterrupt:
        print("\n✅ Test interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return True


def list_available_cameras():
    """List all available cameras."""
    print("🔍 Scanning for available cameras...")
    available_cameras = []

    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()

    if available_cameras:
        print(f"📷 Found cameras at indices: {available_cameras}")
    else:
        print("❌ No cameras found!")

    return available_cameras


def main():
    """Main function for camera testing."""
    print("🎥 Camera Object Detection - Setup Test")
    print("=" * 40)

    # Check OpenCV installation
    try:
        print(f"✅ OpenCV version: {cv2.__version__}")
    except Exception as e:
        print(f"❌ OpenCV not properly installed: {e}")
        return

    # List available cameras
    cameras = list_available_cameras()

    if not cameras:
        print("\n❌ No cameras available. Please check your camera connections.")
        return

    # Test the first available camera
    camera_to_test = cameras[0]
    print(f"\n🎬 Testing camera {camera_to_test}...")

    if test_camera(camera_to_test):
        print("\n🎉 Camera test successful!")
        print("✅ You can now run the full object detection: python main.py")
    else:
        print("\n❌ Camera test failed!")

        # Suggest trying other cameras if available
        if len(cameras) > 1:
            print(f"💡 Try other cameras: {cameras[1:]}")
            print("   Example: python demo.py --camera 1")


if __name__ == "__main__":
    # Simple argument parsing for camera selection
    camera_index = 0
    if len(sys.argv) > 2 and sys.argv[1] == "--camera":
        try:
            camera_index = int(sys.argv[2])
            print(f"Using camera index: {camera_index}")
            test_camera(camera_index)
        except ValueError:
            print("Invalid camera index. Using default camera 0.")
            main()
    else:
        main()
