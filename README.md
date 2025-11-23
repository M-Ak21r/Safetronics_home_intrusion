# Safetronics Home Intrusion Detection System

A real-time computer vision project built for Raspberry Pi to detect home intrusion, security breaches, and theft using advanced object tracking and face detection.

## Features

### 🎯 Real-Time Theft Detection
- **YOLOv8 Object Tracking**: Tracks multiple objects simultaneously with unique IDs
- **Person Filtering**: Automatically ignores objects being held or overlapping with detected persons
- **Displacement Detection**: Triggers alerts when objects move beyond a threshold distance
- **Disappearance Detection**: Detects when tracked objects suddenly disappear from view

### 👤 Face Detection
- **Dual Detection Methods**: 
  - DNN-based face detection (high accuracy)
  - Haar Cascade fallback (lightweight)
- **Proximity-Based Triggering**: Only logs theft events when faces are detected nearby
- **Face Cropping**: Automatically saves cropped face images as evidence

### 📊 Evidence & Logging
- **Annotated Frames**: Saves full frames with bounding boxes and tracking information
- **Face Crops**: Stores individual face images from theft events
- **JSON Logs**: Comprehensive event logs with timestamps, object info, and file references
- **Organized Storage**: Separate directories for logs, faces, and frames

## System Requirements

### Hardware
- Raspberry Pi 4 (4GB+ RAM recommended) or any Linux/Windows/Mac computer
- USB Camera or Raspberry Pi Camera Module
- Minimum 8GB storage for evidence and models

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
  --model yolov8n.pt \
  --displacement-threshold 150 \
  --disappearance-frames 45 \
  --face-method dnn \
  --confidence 0.6
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--source` | Video source (camera index or file path) | 0 |
| `--model` | YOLOv8 model path | yolov8n.pt |
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
CONFIDENCE_THRESHOLD = 0.5     # YOLO confidence

# Face detection
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar"
FACE_CONFIDENCE_THRESHOLD = 0.7

# Model paths
YOLO_MODEL = "yolov8n.pt"
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
Two conditions trigger theft alerts:

**Displacement**: Object moves beyond threshold distance
- Tracks object center point over time
- Calculates total displacement from first detection
- Triggers when displacement > threshold (default: 100px)

**Disappearance**: Object missing for extended period
- Counts frames where tracked object is not detected
- Triggers when missing_frames > threshold (default: 30 frames)

### 4. Face Detection & Evidence
- When theft detected, system searches for nearby faces
- Only logs event if face(s) detected (indicates human involvement)
- Saves:
  - Annotated frame showing the scene
  - Cropped face images
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

### For Raspberry Pi
- Use YOLOv8 nano model (`yolov8n.pt`) for better performance
- Reduce frame resolution in config.py
- Use Haar Cascade face detection instead of DNN
- Increase face detection interval (`FACE_DETECTION_INTERVAL`)

### For Desktop/Server
- Use larger YOLOv8 models (`yolov8s.pt`, `yolov8m.pt`) for better accuracy
- Use DNN face detection for higher accuracy
- Increase frame resolution for better detection

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
