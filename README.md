# Safetronics Home Intrusion Detection System

A real-time computer vision project for detecting theft using stationary object monitoring and hand-based theft detection.

## Features

### 🎯 Stationary Object Monitoring
- **YOLOv8 Object Detection**: Uses YOLOv8m (medium model) for balanced accuracy and speed
- **Stationary Detection**: Objects become monitored after remaining stable for a configurable period
- **Hitbox Creation**: Expanded detection zones (hitboxes) around stationary objects
- **Continuous Monitoring**: Tracks all detected objects in the scene

### ✋ Hand-Based Theft Detection
- **Hand Region Estimation**: Estimates hand positions based on person bounding boxes
- **Hitbox Intrusion Detection**: Triggers alerts when a person's hand enters an object's hitbox
- **Immediate Response**: Detects theft attempts in real-time
- **Person Identification**: Tracks and marks persons who attempt theft

### 👤 Person Tracking
- **Dual Detection Methods**: 
  - DNN-based face detection (high accuracy)
  - Haar Cascade fallback (lightweight)
- **Face Cropping**: Automatically saves cropped face images as evidence
- **Thief Marking**: Marks persons as thieves when their hand enters a hitbox
- **Evidence Collection**: Captures person images at moment of theft attempt

### ⚡ Multithreading Support
- **Parallel Frame Processing**: Separate threads for capture and processing
- **Thread-Safe Frame Buffer**: Efficient frame queue management
- **Non-Blocking Display**: Smooth video display even during heavy processing
- **Configurable Queue Size**: Adjustable buffer for different hardware capabilities

### 📊 Evidence & Logging
- **Annotated Frames**: Saves full frames with bounding boxes, hitboxes, and alerts
- **Face Crops**: Stores individual face images from theft events
- **Person Captures**: Saves images of persons who attempted theft
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
CONFIDENCE_THRESHOLD = 0.5     # YOLO confidence threshold

# Stationary object settings
STATIONARY_THRESHOLD = 5       # pixels - max movement to be considered stationary
STATIONARY_FRAMES = 30         # frames - how long object must be stable
HITBOX_MARGIN = 50             # pixels - expansion margin around object

# Face detection
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar"
FACE_CONFIDENCE_THRESHOLD = 0.7

# Model paths
YOLO_MODEL = "yolov8m.pt"      # Medium model - balanced accuracy and speed

# Display settings
DRAW_HITBOXES = True           # Show hitboxes around stationary objects
DRAW_HAND_REGIONS = False      # Show estimated hand regions (for debugging)

# Multithreading settings
ENABLE_MULTITHREADING = True   # Enable for faster processing
FRAME_QUEUE_SIZE = 8           # Size of frame buffer
```

## How It Works

### 1. Object Detection & Tracking
- YOLOv8 detects objects in each frame
- Each object receives a unique tracking ID
- Objects are monitored for stability

### 2. Stationary Object Detection
- System tracks object movement over time
- When object remains stable (< 5px movement) for 30 frames, it's marked as "stationary"
- A hitbox (expanded detection zone) is created around stationary objects
- Hitbox margin is configurable (default: 50px expansion)

### 3. Hand Detection
- System tracks all persons in the frame
- Hand regions are estimated based on person bounding box
- Hand positions are approximated at ~40-75% of person height, extending beyond body width

### 4. Theft Detection
**Hand-in-Hitbox Detection**:
- Continuously checks if any person's hand region overlaps with a stationary object's hitbox
- If overlap is detected → **THEFT ATTEMPT**
- Person is immediately marked as "thief"
- Alert is displayed on screen

### 5. Evidence Collection
- When theft detected:
  - Annotated frame is saved showing the scene
  - Faces are detected and cropped
  - Person images are captured
  - JSON log is created with all metadata
- Evidence includes:
  - Object hitbox location
  - Person ID involved
  - Timestamp
  - Face/person images

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
