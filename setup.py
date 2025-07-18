#!/usr/bin/env python3
"""
Setup script for the Camera Object Detection project.

This script helps users set up their environment and install dependencies.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed!")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False

    # Install dependencies
    commands = [
        ("pip install --upgrade pip", "Upgrading pip"),
        ("pip install -r requirements.txt", "Installing project dependencies"),
    ]

    for command, description in commands:
        if not run_command(command, description):
            return False

    return True


def test_imports():
    """Test if all required packages can be imported."""
    print("\n🧪 Testing package imports...")

    packages = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("ultralytics", "Ultralytics YOLOv8"),
        ("torch", "PyTorch"),
        ("argparse", "argparse"),
        ("time", "time"),
        ("typing", "typing"),
    ]

    all_good = True

    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {name}: {e}")
            all_good = False

    return all_good


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating project directories...")

    directories = ["screenshots", "output", "models"]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")


def test_camera_basic():
    """Basic camera test without OpenCV GUI."""
    print("\n📹 Testing camera access...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access successful")
                cap.release()
                return True
            else:
                print("❌ Camera opened but couldn't read frame")
        else:
            print("❌ Could not open camera")

        cap.release()
        return False

    except ImportError:
        print("❌ OpenCV not available for camera test")
        return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🎥 Camera Object Detection - Setup Script")
    print("=" * 45)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        print("Please install Python 3.8 or higher")
        return

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        print("Please check your internet connection and try again")
        return

    # Test imports
    if not test_imports():
        print("\n❌ Setup failed: Some packages could not be imported")
        print("Try running: pip install -r requirements.txt")
        return

    # Create directories
    create_directories()

    # Test camera
    test_camera_basic()

    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run 'python demo.py' to test your camera")
    print("2. Run 'python main.py' to start object detection")
    print("3. Press 'q' to quit any application")
    print("4. Press 'space' to take screenshots")

    print("\n💡 Useful commands:")
    print("- python main.py --help          (show help)")
    print("- python main.py --camera 1      (use different camera)")
    print("- python main.py --confidence 0.3 (lower detection threshold)")


if __name__ == "__main__":
    main()
