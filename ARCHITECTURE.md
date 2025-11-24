# System Architecture

This document describes the architecture of the Safetronics Home Intrusion Detection System.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Video Input Source                           │
│              (Webcam / USB Camera / Video File)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   TheftDetector (Main Engine)                    │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          YOLOv8 Object Detection & Tracking             │   │
│  │  - Detect objects in each frame                         │   │
│  │  - Track objects with unique IDs                        │   │
│  │  - Identify person boxes                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Person Overlap Filtering                   │   │
│  │  - Calculate IoU between objects and persons            │   │
│  │  - Filter out objects held by people                    │   │
│  │  - Mark filtered objects (not tracked for theft)        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Object State Management                     │   │
│  │  - Update TrackedObject instances                       │   │
│  │  - Record position history                              │   │
│  │  - Calculate displacement                               │   │
│  │  - Track missing frames                                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Theft Condition Detection                   │   │
│  │  - Check displacement > threshold                       │   │
│  │  - Check missing_frames > threshold                     │   │
│  │  - Trigger theft event if conditions met                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Face Detection (DNN/Haar)                   │   │
│  │  - Detect faces near theft location                     │   │
│  │  - Only proceed if faces detected                       │   │
│  │  - Crop face regions                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Evidence & Logging                          │   │
│  │  - Save annotated frame                                 │   │
│  │  - Save face crops                                      │   │
│  │  - Write JSONL log entry                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Output & Storage                          │
│                                                                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐     │
│  │   Display   │  │   Logs      │  │   Evidence Files    │     │
│  │   Window    │  │   (JSONL)   │  │  - Frames           │     │
│  │             │  │             │  │  - Face Crops       │     │
│  └─────────────┘  └─────────────┘  └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Video Input
- **Source**: Camera device or video file
- **Processing**: OpenCV VideoCapture
- **Frame Rate**: Configurable (default 30 FPS)

### 2. YOLOv8 Detection & Tracking
- **Model**: YOLOv8 (nano by default for Raspberry Pi)
- **Purpose**: Detect and track objects across frames
- **Output**: Bounding boxes, class labels, confidence scores, track IDs

### 3. Person Filtering
- **Algorithm**: Intersection over Union (IoU)
- **Threshold**: 0.5 (configurable)
- **Purpose**: Ignore objects being carried or held by people
- **Implementation**: `utils/geometry.py::calculate_iou()`

### 4. Object Tracking
- **Data Structure**: `TrackedObject` class
- **State**:
  - Position history (last 100 positions)
  - Total displacement
  - Missing frame counter
  - First/last seen timestamps
- **Purpose**: Maintain temporal continuity of objects

### 5. Theft Detection Logic

#### Displacement Detection
```python
if object.get_displacement() > DISPLACEMENT_THRESHOLD:
    trigger_theft_event("displacement")
```

#### Disappearance Detection
```python
if object.missing_frames > DISAPPEARANCE_FRAMES:
    trigger_theft_event("disappearance")
```

### 6. Face Detection
- **Methods**:
  - **DNN**: Caffe-based SSD face detector (accurate)
  - **Haar**: Cascade classifier (fast, built-in)
- **Proximity Check**: Only faces within radius of theft location
- **Purpose**: Confirm human involvement in theft

### 7. Evidence Logging
- **Format**: JSONL (JSON Lines) for efficient append-only writes
- **Components**:
  - Event metadata (timestamp, type, object info)
  - Annotated frame image
  - Face crop images
  - File references

## Data Flow

```
Frame Input → YOLOv8 Detection → Track Assignment → Person Filtering
                                                            ↓
   Evidence ← Face Detection ← Theft Detection ← Position Update
```

## Key Algorithms

### IoU Calculation
```
IoU = Intersection Area / Union Area
```
Used to determine if object overlaps with person box.

### Displacement Calculation
```
Displacement = √[(x₂-x₁)² + (y₂-y₁)²]
```
Euclidean distance between first and current position.

### Face Proximity Check
```
Distance = √[(face_center_x - object_center_x)² + 
            (face_center_y - object_center_y)²]

if Distance < SEARCH_RADIUS:
    faces_nearby.append(face)
```

## Module Structure

```
safetronics_home_intrusion/
├── main.py                      # CLI entry point
├── theft_detector.py            # Main detection engine
│   ├── TheftDetector            # Main class
│   └── TrackedObject            # Object state
├── config.py                    # Configuration
└── utils/
    ├── face_detector.py         # Face detection
    │   └── FaceDetector
    ├── geometry.py              # Geometric calculations
    │   ├── calculate_iou()
    │   ├── calculate_center()
    │   ├── calculate_distance()
    │   └── box format converters
    └── logger.py                # Event logging
        └── TheftLogger
```

## Configuration Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `DISPLACEMENT_THRESHOLD` | 100px | Minimum movement to trigger alert |
| `DISAPPEARANCE_FRAMES` | 30 | Frames before disappearance alert |
| `CONFIDENCE_THRESHOLD` | 0.5 | YOLO detection confidence |
| `IOU_THRESHOLD` | 0.5 | Person overlap threshold |
| `FACE_CONFIDENCE_THRESHOLD` | 0.7 | Face detection confidence |
| `FACE_DETECTION_INTERVAL` | 3 | Detect faces every N frames |
| `MAX_POSITION_HISTORY` | 100 | Max positions to store |

## Performance Considerations

### Raspberry Pi Optimization
1. **YOLOv8 Nano**: Smallest model for faster inference
2. **Lower Resolution**: 640x480 default
3. **Haar Cascade**: Faster than DNN face detection
4. **Face Detection Interval**: Every 3rd frame
5. **Position History Limit**: Cap at 100 positions

### Desktop/Server Optimization
1. **Larger YOLO Models**: yolov8s, yolov8m for accuracy
2. **Higher Resolution**: 1280x720 or 1920x1080
3. **DNN Face Detection**: Better accuracy
4. **Lower Detection Interval**: Every frame

## Security & Privacy

### Data Handling
- Evidence stored locally only
- No cloud upload by default
- Face crops stored separately for review
- JSONL logs for audit trail

### Configurable Privacy
- Disable face detection: Set `FACE_DETECTION_INTERVAL = 0`
- Disable evidence saving: Modify logger calls
- Auto-cleanup: Implement retention policies

## Extensibility

### Adding New Detection Types
```python
def _check_custom_condition(self, obj, frame):
    if obj.custom_metric > threshold:
        return "custom_event"
    return None
```

### Custom Face Detection
```python
class CustomFaceDetector(FaceDetector):
    def detect_faces(self, frame):
        # Custom implementation
        return faces
```

### Additional Logging
```python
logger.log_custom_event(event_type, data, frame)
```

## Error Handling

1. **Camera Failure**: Graceful exit with error message
2. **Model Loading**: Fallback to built-in models
3. **Face Detection Failure**: Continue without faces
4. **Logging Errors**: Print warning, continue operation
5. **Invalid Detections**: Skip and continue

## Testing Strategy

### Unit Tests
- Geometry calculations
- Face detector initialization
- Logger file operations
- Object tracking state

### Integration Tests
- End-to-end frame processing
- Theft detection triggers
- Evidence generation

### Performance Tests
- FPS measurement
- Memory usage monitoring
- Detection accuracy

## Future Enhancements

1. **Multi-camera support**: Track across cameras
2. **Alert notifications**: Email, SMS, push
3. **Web dashboard**: Real-time monitoring UI
4. **Person recognition**: Identify known vs unknown
5. **Cloud backup**: Optional evidence upload
6. **ML improvements**: Custom trained models
7. **Zone configuration**: Define sensitive areas
8. **Scheduled recording**: Time-based activation

## References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Detection Paper](https://arxiv.org/abs/1708.05234)
