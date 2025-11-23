"""
Configuration file for theft detection system
"""

# Detection thresholds
DISPLACEMENT_THRESHOLD = 100  # pixels - minimum displacement to trigger theft alert
DISAPPEARANCE_FRAMES = 30  # frames - object must be missing for this many frames
CONFIDENCE_THRESHOLD = 0.5  # YOLOv8 detection confidence threshold
IOU_THRESHOLD = 0.5  # IoU threshold for overlap detection with person boxes

# Face detection
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar"
FACE_CONFIDENCE_THRESHOLD = 0.7  # DNN face detection confidence
FACE_SCALE_FACTOR = 1.1  # Haar cascade scale factor
FACE_MIN_NEIGHBORS = 5  # Haar cascade min neighbors
FACE_DETECTION_INTERVAL = 3  # Detect faces every N frames for performance

# Model paths
YOLO_MODEL = "yolov8n.pt"  # YOLOv8 nano model (lightweight for Raspberry Pi)
DNN_PROTO = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
HAAR_CASCADE = "models/haarcascade_frontalface_default.xml"

# Video settings
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Logging
LOG_DIR = "logs"
EVIDENCE_DIR = "evidence"
FACE_DIR = "evidence/faces"
FRAME_DIR = "evidence/frames"
MAX_LOG_FILES = 100  # Maximum number of log files to keep

# Display
SHOW_DISPLAY = True  # Show real-time video feed
DRAW_BOXES = True  # Draw bounding boxes on display
DRAW_TRACKS = True  # Draw object tracks

# Object filtering
IGNORE_CLASSES = ["person"]  # Classes to ignore for theft detection
TRACKED_CLASSES = []  # Empty means track all non-person objects
