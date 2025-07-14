# Camera Object Detection with YOLOv8

A real-time object detection application using Python, OpenCV, and YOLOv8 from Ultralytics that captures video from your computer's camera and detects objects in the live feed.

![Object Detection Demo](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **YOLOv8 Integration**: State-of-the-art object detection with Ultralytics YOLOv8
- **Real-time Detection**: Live object detection from camera feed with high accuracy
- **Multiple Model Sizes**: Support for YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- **GPU Acceleration**: CUDA and MPS support for faster inference
- **80+ Object Classes**: Supports full COCO dataset object classes
- **Visual Feedback**: Bounding boxes, confidence scores, and detection counts
- **Performance Monitoring**: FPS counter and inference time display
- **Screenshot Capture**: Save images with detected objects
- **Configurable Settings**: Adjustable detection thresholds and parameters
- **Cross-platform**: Works on Windows, macOS, and Linux

## Quick Start

### Prerequisites

- Python 3.8 or higher
- A working camera (built-in or USB)
- Git (for cloning the repository)

### Installation

1. **Clone or download this project**
   ```bash
   cd "d:\Documents\Object Detect"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

### Basic Usage

1. **Start the application**: Run `python main.py`
2. **Camera will open**: You'll see a live video feed from your camera
3. **Objects will be detected**: Detected objects will have bounding boxes and labels
4. **Take screenshots**: Press `Space` to save a screenshot
5. **Exit**: Press `Q` to quit the application

## Configuration

You can customize the detection behavior by modifying `config.py`:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
NMS_THRESHOLD = 0.4         # Non-maximum suppression threshold

# Camera settings
CAMERA_WIDTH = 640          # Camera resolution width
CAMERA_HEIGHT = 480         # Camera resolution height

# Display settings
SHOW_FPS = True            # Show FPS counter
SHOW_CONFIDENCE = True     # Show confidence scores
```

## Command Line Options

```bash
python main.py --help
```

Available options:
- `--camera INDEX`: Choose camera index (default: 0)
- `--model MODEL`: YOLOv8 model to use (default: yolov8n.pt)
- `--confidence FLOAT`: Set confidence threshold (default: 0.5)
- `--device DEVICE`: Device to run inference on (default: cpu)
- `--imgsz SIZE`: Input image size (default: 640)
- `--half`: Use half precision (FP16) for faster inference
- `--verbose`: Show detailed model loading info

Examples:
```bash
# Use external USB camera
python main.py --camera 1

# Use larger YOLOv8 model for better accuracy
python main.py --model yolov8s.pt

# Lower confidence threshold for more detections
python main.py --confidence 0.3

# Use GPU acceleration (if CUDA available)
python main.py --device cuda

# Combine multiple options
python main.py --camera 1 --model yolov8m.pt --confidence 0.6 --device cuda
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `Space` | Take screenshot |
| `ESC` | Alternative quit method |

## Project Structure

```
Object Detect/
├── main.py                 # Main application file
├── config.py              # Configuration settings
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
├── .github/
│   └── copilot-instructions.md  # Copilot development guidelines
├── screenshots/           # Saved screenshots (created automatically)
├── output/               # Output files (created automatically)
└── models/               # Model files (created automatically)
```

## Detected Object Classes

The application can detect 80+ object classes from the COCO dataset, including:

**People & Animals**: person, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, bird

**Vehicles**: car, motorcycle, airplane, bus, train, truck, boat, bicycle

**Household Items**: chair, couch, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone

**Food Items**: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

And many more! See `main.py` for the complete list.

## Troubleshooting

### Camera Issues

**Problem**: "Error: Could not open camera"
- **Solution**: Check if your camera is being used by another application
- **Alternative**: Try different camera index: `python main.py --camera 1`

**Problem**: Poor detection performance
- **Solution**: Ensure good lighting and try adjusting confidence threshold
- **Example**: `python main.py --confidence 0.3`

### Performance Issues

**Problem**: Low FPS or laggy video
- **Solutions**:
  - Close other applications using the camera
  - Lower the camera resolution in `config.py`
  - Increase `SKIP_FRAMES` in `config.py`

### Installation Issues

**Problem**: "cv2 module not found"
- **Solution**: Install OpenCV: `pip install opencv-python`

**Problem**: "numpy module not found"
- **Solution**: Install NumPy: `pip install numpy`

## Advanced Features

### Enhanced Object Detection

For more accurate detection, you can integrate advanced models:

1. **YOLO Integration**: Download YOLO weights and config files
2. **TensorFlow Models**: Use TensorFlow Object Detection API
3. **Custom Models**: Train your own models for specific use cases

### Recording Video

Modify `config.py` to enable video recording:

```python
ENABLE_RECORDING = True
OUTPUT_VIDEO_PATH = "output/detection_output.avi"
```

### Web Interface

Consider adding a web interface using Flask or Streamlit for remote access and better user experience.

## Development

### Setting Up Development Environment

1. **Install development dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest black flake8  # Additional dev tools
   ```

2. **Code formatting**:
   ```bash
   black *.py  # Format code
   flake8 *.py  # Check code style
   ```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Adding New Features

- **New detection models**: Extend the `ObjectDetector` class in `main.py`
- **UI improvements**: Modify the drawing functions in `main.py`
- **Configuration options**: Add settings to `config.py`
- **Utility functions**: Add helpers to `utils.py`

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, or Ubuntu 18.04
- **Python**: 3.8 or higher
- **RAM**: 4GB
- **Camera**: Any USB or built-in camera

### Recommended Requirements
- **OS**: Latest version of Windows, macOS, or Linux
- **Python**: 3.9 or higher
- **RAM**: 8GB or more
- **Camera**: HD camera (720p or higher)
- **CPU**: Multi-core processor for better performance

## FAQ

**Q: Can I use an IP camera or external camera?**
A: Yes, modify the camera initialization in `main.py` to use IP camera streams or specify different camera indices.

**Q: How do I improve detection accuracy?**
A: Use better lighting, ensure objects are clearly visible, and consider integrating more advanced models like YOLO or custom-trained models.

**Q: Can I detect custom objects?**
A: The current version uses pre-trained COCO models. For custom objects, you'll need to train your own model or use transfer learning.

**Q: Does this work on Raspberry Pi?**
A: Yes, but you may need to optimize performance settings and ensure proper camera drivers are installed.

## License

This project is licensed under the MIT License - see the code comments for details.

## Acknowledgments

- **OpenCV Community**: For the excellent computer vision library
- **COCO Dataset**: For providing the object class definitions
- **Python Community**: For the robust ecosystem of libraries

## Support

If you encounter issues or have questions:

1. Check the troubleshooting section above
2. Review the configuration options in `config.py`
3. Ensure all dependencies are properly installed
4. Test with different camera settings

## Version History

- **v1.0.0**: Initial release with basic object detection
  - Real-time camera capture
  - Basic object detection framework
  - Screenshot functionality
  - Configurable settings

---

**Note**: This is a demonstration project that provides a foundation for object detection applications. For production use, consider integrating more advanced detection models and implementing additional security and performance optimizations.
