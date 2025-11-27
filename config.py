"""
Configuration file for theft detection system
Uses hand-in-hitbox detection for stationary objects
"""

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5  # YOLOv8 detection confidence threshold
IOU_THRESHOLD = 0.5  # IoU threshold for overlap detection
NMS_IOU_THRESHOLD = 0.45  # Non-maximum suppression IoU threshold

# Stationary object settings
STATIONARY_THRESHOLD = 5  # pixels - max movement for object to be considered stationary
STATIONARY_FRAMES = 30  # frames - how long object must be stable before being monitored
HITBOX_MARGIN = 50  # pixels - expansion margin around object for hitbox
OBJECT_MISSING_TIMEOUT = 90  # frames - remove object if missing this long

# Face detection (optimized for PC)
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar" - DNN for better accuracy on PC
FACE_CONFIDENCE_THRESHOLD = 0.7  # DNN face detection confidence
FACE_SCALE_FACTOR = 1.1  # Haar cascade scale factor
FACE_MIN_NEIGHBORS = 5  # Haar cascade min neighbors
FACE_DETECTION_INTERVAL = 1  # Detect faces every frame on PC

# Model paths
YOLO_MODEL = "yolov8m.pt"  # YOLOv8 medium model (balanced accuracy and speed)
DNN_PROTO = "models/deploy.prototxt"
DNN_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
HAAR_CASCADE = "models/haarcascade_frontalface_default.xml"

# Video settings (optimized for PC)
VIDEO_SOURCE = 0  # 0 for webcam, or path to video file
FRAME_WIDTH = 1280  # Higher resolution for PC
FRAME_HEIGHT = 720  # Higher resolution for PC
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
DRAW_TRACKS = False  # Draw object tracks (disabled for new approach)
DRAW_HITBOXES = True  # Draw hitboxes around stationary objects
DRAW_HAND_REGIONS = False  # Draw estimated hand regions (for debugging)

# Object filtering
IGNORE_CLASSES = ["person"]  # Classes to ignore for theft detection
TRACKED_CLASSES = []  # Empty means track all non-person objects

# Person tracking settings
MAX_POSITION_HISTORY = 100  # Maximum number of positions to keep in history
THIEF_PROXIMITY_THRESHOLD = 200  # pixels - distance threshold for person near object
THIEF_STATUS_TIMEOUT = 300  # frames - how long to keep thief status (10 sec at 30fps)
PERSON_MISSING_TIMEOUT = 90  # frames - remove tracked person if missing this long

# Multithreading settings
ENABLE_MULTITHREADING = True  # Enable multithreading for faster processing
FRAME_QUEUE_SIZE = 8  # Size of frame buffer queue
PROCESSING_THREADS = 2  # Number of processing threads

# Display colors (BGR format)
COLOR_PERSON = (0, 255, 0)  # Green for normal persons
COLOR_OBJECT = (255, 0, 0)  # Blue for objects being tracked
COLOR_STATIONARY = (255, 255, 0)  # Cyan for stationary objects
COLOR_THIEF = (0, 0, 255)  # Red for persons marked as thieves
COLOR_HITBOX = (0, 255, 255)  # Yellow for hitboxes
COLOR_FILTERED = (128, 128, 128)  # Gray for filtered objects
