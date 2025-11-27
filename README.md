# Safetronics Home Intrusion Detection System

A real-time computer vision project built for Raspberry Pi to detect home intrusion, security breaches, and theft using advanced object tracking and face detection.

## Features

### 🎯 Real-Time Theft Detection
- **YOLOv8 Object Tracking**: Tracks multiple objects simultaneously with unique IDs using YOLOv8m (medium model) for balanced accuracy and speed
- **Person Filtering**: Automatically ignores objects being held or overlapping with detected persons
- **Displacement Detection**: Triggers alerts when objects move beyond a threshold distance
- **Disappearance Detection**: Detects when tracked objects suddenly disappear from view
- **Out-of-Scope Detection**: Marks objects as stolen when they leave the camera view

### 🔄 Optimized Tracking
- **Velocity-Based Prediction**: Predicts object positions during brief occlusions to maintain tracking
- **Custom ByteTrack Configuration**: Optimized tracker settings for persistent tracking
- **Predicted Box Display**: Shows dashed orange boxes for temporarily lost objects
- **Extended Track Buffer**: Maintains object identity during longer occlusions

### 👤 Face & Person Detection
- **Dual Detection Methods**: 
  - DNN-based face detection (high accuracy)
  - Haar Cascade fallback (lightweight)
- **Proximity-Based Triggering**: Captures persons near tracked objects
- **Face Cropping**: Automatically saves cropped face images as evidence
- **Thief Marking**: Marks persons as thieves when nearby objects disappear

### ⚡ Multithreading Support
- **Parallel Frame Processing**: Separate threads for capture and processing
- **Thread-Safe Frame Buffer**: Efficient frame queue management
- **Non-Blocking Display**: Smooth video display even during heavy processing
- **Configurable Queue Size**: Adjustable buffer for different hardware capabilities

### 📊 Evidence & Logging
- **Annotated Frames**: Saves full frames with bounding boxes and tracking information
- **Face Crops**: Stores individual face images from theft events
- **Person Captures**: Saves images of persons who were near objects
- **JSON Logs**: Comprehensive event logs with timestamps, object info, and file references
- **Organized Storage**: Separate directories for logs, faces, and frames

## System Requirements

### Hardware
- PC/Desktop with Windows/Linux/Mac (optimized for desktop performance)
- USB Webcam or built-in camera
- Minimum 8GB storage for evidence and models
- GPU recommended for best performance (CUDA-compatible for NVIDIA GPUs)

### Software
- Python 3.8 or higher
- OpenCV 4.8+
- YOLOv8 (ultralytics)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/M-Ak21r/Safetronics_home_intrusion.git
cd Safetronics_home_intrusion
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Face Detection Models (Optional but Recommended)
For better face detection accuracy, download the DNN models:

```bash
cd models

# Download DNN prototxt
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt

# Download DNN model weights
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

cd ..
```

If DNN models are not available, the system will automatically fall back to Haar Cascade detection.

## Usage

### Basic Usage (Webcam)
```bash
python main.py
```

### Use a Video File
```bash
python main.py --source path/to/video.mp4
```

### Use Different Camera
```bash
python main.py --source 1
```

### Advanced Options
```bash
python main.py \
  --source 0 \
  --model yolov8m.pt \
  --displacement-threshold 150 \
  --disappearance-frames 45 \
  --face-method dnn \
  --confidence 0.5
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Video source (camera index or file path) | 0 |
| `--model` | YOLOv8 model path | yolov8m.pt |
| `--displacement-threshold` | Displacement threshold in pixels | 100 |
| `--disappearance-frames` | Frames before disappearance alert | 30 |
| `--face-method` | Face detection method (dnn/haar) | dnn |
| `--no-display` | Disable video display window | False |
| `--confidence` | YOLO confidence threshold | 0.5 |

### Controls
- Press `q` to quit the application
- Press `Ctrl+C` to interrupt

## Configuration

Edit `config.py` to customize detection parameters:

```python
# Detection thresholds
DISPLACEMENT_THRESHOLD = 100  # pixels
DISAPPEARANCE_FRAMES = 30      # frames
CONFIDENCE_THRESHOLD = 0.5     # YOLO confidence (optimized for YOLOv8m)

# Tracking optimization settings
TRACK_BUFFER = 60             # frames to keep tracking lost objects
PREDICTION_FRAMES = 15        # frames to predict object position
OUT_OF_SCOPE_FRAMES = 45      # frames before marking as theft (out of scope)

# Face detection
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar"
FACE_CONFIDENCE_THRESHOLD = 0.7

# Model paths
YOLO_MODEL = "yolov8m.pt"  # Medium model - balanced accuracy and speed

# Multithreading settings
ENABLE_MULTITHREADING = True  # Enable for faster processing
FRAME_QUEUE_SIZE = 8          # Size of frame buffer
```

## How It Works

### 1. Object Detection & Tracking
- YOLOv8 detects and tracks objects in each frame
- Each object receives a unique tracking ID
- Object positions are recorded over time

### 2. Person Filtering
- System identifies all "person" detections
- Calculates IoU (Intersection over Union) between objects and person boxes
- Objects overlapping with persons (IoU > 0.5) are filtered out
- Prevents false alarms from people carrying items

### 3. Theft Detection
Three conditions trigger theft alerts:

**Displacement**: Object moves beyond threshold distance
- Tracks object center point over time
- Calculates total displacement from first detection
- Triggers when displacement > threshold (default: 100px)

**Disappearance**: Object missing for extended period
- Counts frames where tracked object is not detected
- Uses velocity prediction to maintain tracking during brief occlusions
- Triggers when missing_frames > threshold (default: 30 frames)

**Out-of-Scope**: Object leaves camera view
- Predicts object trajectory using velocity
- Detects when predicted position exits frame boundaries
- Marks nearby persons as potential thieves

### 4. Face Detection & Evidence
- When theft detected, system searches for nearby faces
- Captures images of persons who were near the object
- Logs event with comprehensive evidence
- Saves:
  - Annotated frame showing the scene
  - Cropped face images
  - Person captures from nearby persons
  - JSON log with all metadata

## Project Structure

```
Safetronics_home_intrusion/
├── main.py                 # Main application entry point
├── theft_detector.py       # Core theft detection logic
├── config.py              # Configuration parameters
├── requirements.txt       # Python dependencies
├── utils/
│   ├── __init__.py
│   ├── face_detector.py   # Face detection module
│   ├── geometry.py        # Geometric calculations (IoU, distance)
│   └── logger.py          # Event logging and evidence saving
├── models/                # Model files (YOLOv8, DNN)
│   └── README.md
├── logs/                  # JSON event logs
│   └── README.md
└── evidence/             # Saved evidence
    ├── faces/            # Cropped face images
    ├── frames/           # Annotated frames
    └── README.md
```

## Output Examples

### Console Output
```
Theft Detection System Started
============================================================
Press 'q' to quit

Frame: 245 | Objects: 3 | Persons: 1
⚠️  THEFT DETECTED: displacement - backpack #42 (displacement: 156.3px, faces: 2)
```

### JSON Log Entry
```json
{
  "event_id": "20231123_143022_123456",
  "timestamp": "2023-11-23T14:30:22.123456",
  "event_type": "displacement",
  "object": {
    "track_id": 42,
    "class_name": "backpack",
    "confidence": 0.89,
    "bbox": [100, 150, 300, 350],
    "displacement": 156.3
  },
  "faces_detected": 2,
  "evidence": {
    "frame": "evidence/frames/frame_20231123_143022_123456.jpg",
    "faces": [
      "evidence/faces/face_20231123_143022_123456_0.jpg",
      "evidence/faces/face_20231123_143022_123456_1.jpg"
    ]
  }
}
```

## Performance Tips

### Current Configuration (Optimized for PC with Multithreading)
The system is now optimized for PC/Desktop use with balanced performance:
- **YOLOv8 medium model (`yolov8m.pt`)** - Good balance of accuracy and speed
- **Multithreading enabled** - Separate capture and processing threads
- 1280x720 resolution for detailed detection
- DNN face detection for maximum accuracy
- Face detection every frame
- Confidence threshold: 0.5
- Velocity-based tracking prediction

**Note:** YOLOv8m provides a good balance between detection accuracy and processing speed. Multithreading enables smooth operation even on systems without dedicated GPU.

### For High-End Systems
To maximize accuracy with GPU acceleration:
- Use YOLOv8 extra-large model (`yolov8x.pt`) for highest accuracy
- Increase frame resolution to 1920x1080
- Keep multithreading enabled

### For Lower-End Systems
To optimize for systems without GPU or Raspberry Pi, modify `config.py`:
- Use YOLOv8 small model (`yolov8s.pt`) or nano (`yolov8n.pt`) for better performance
- Set `ENABLE_MULTITHREADING = False` if experiencing issues
- Reduce frame resolution to 640x480
- Use Haar Cascade face detection instead of DNN
- Increase face detection interval to 3 frames (`FACE_DETECTION_INTERVAL = 3`)

## Troubleshooting

### "Could not open video source"
- Check camera index (try 0, 1, 2)
- Ensure camera is connected and not in use
- On Linux, check permissions: `sudo chmod 666 /dev/video0`

### "DNN model files not found"
- Download models as described in Installation section
- Or use Haar Cascade: `--face-method haar`

### Poor Detection Performance
- Adjust `--confidence` threshold (lower = more detections)
- Try different YOLOv8 model sizes
- Ensure good lighting conditions

### False Positives
- Increase `--displacement-threshold`
- Increase `--disappearance-frames`
- Adjust IoU threshold in config.py

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection and tracking
- [OpenCV](https://opencv.org/) - Computer vision library
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv) - Face detection models

## Contact

For questions and support, please open an issue on GitHub.
