<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Camera Object Detection Project Instructions

This is a Python-based computer vision project that uses OpenCV for real-time object detection through a camera feed.

## Project Structure
- `main.py`: Main application with camera capture and object detection
- `config.py`: Configuration settings for detection parameters
- `utils.py`: Utility functions for image processing and file operations
- `requirements.txt`: Python dependencies

## Key Technologies
- **OpenCV**: Computer vision library for camera access and image processing
- **YOLOv8 (Ultralytics)**: Object detection, instance segmentation, pose, and tracking
- **NumPy**: Numerical computing for array operations
- **Python 3.8+**: Core programming language

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Include comprehensive docstrings for all classes and functions
- Use descriptive variable names, especially for computer vision operations

### Computer Vision Best Practices
- Always validate camera initialization before proceeding
- Handle frame capture failures gracefully
- Implement proper resource cleanup (camera release, window destruction)
- Use appropriate color space conversions (BGR to RGB, etc.)
- Consider performance implications of image processing operations

### Object Detection Guidelines
- Implement confidence thresholds for filtering detections
- Use Non-Maximum Suppression (NMS) to reduce duplicate detections
- Provide clear visual feedback with bounding boxes and labels
- Allow configuration of detection parameters through config.py

### Error Handling
- Wrap camera operations in try-catch blocks
- Provide informative error messages for common issues (camera not found, etc.)
- Implement fallback mechanisms for missing dependencies
- Validate input parameters before processing

### Performance Considerations
- Implement FPS monitoring and display
- Allow frame skipping for better performance on slower systems
- Provide options for frame resizing to improve processing speed
- Consider multi-threading for camera capture and processing separation

### User Experience
- Display helpful on-screen instructions
- Implement keyboard shortcuts for common actions
- Provide visual feedback for all user interactions
- Save screenshots and detection results with timestamps

## Testing Notes
- Test with different camera resolutions and frame rates
- Verify detection accuracy with various objects and lighting conditions
- Ensure proper cleanup when application is terminated
- Test keyboard shortcuts and user interactions

## Future Enhancements
- Integration with YOLO or other advanced detection models
- Support for video file input in addition to live camera
- Web interface using Flask or Streamlit
- Object tracking between frames
- Recording and playback functionality
