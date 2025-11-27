"""
Configuration file for theft detection system
"""

# Detection thresholds
DISPLACEMENT_THRESHOLD = 100  # pixels - minimum displacement to trigger theft alert
DISAPPEARANCE_FRAMES = 30  # frames - object must be missing for this many frames
CONFIDENCE_THRESHOLD = 0.5  # YOLOv8 detection confidence threshold (optimized for YOLOv8m)
IOU_THRESHOLD = 0.5  # IoU threshold for overlap detection with person boxes
NMS_IOU_THRESHOLD = 0.45  # Non-maximum suppression IoU threshold for better precision

# Tracking optimization settings
TRACK_BUFFER = 60  # frames - how long to keep tracking objects that are temporarily lost
TRACK_HIGH_THRESH = 0.5  # High detection threshold for new tracks
TRACK_LOW_THRESH = 0.1  # Low detection threshold for matching existing tracks
TRACK_MATCH_THRESH = 0.8  # Matching threshold for track association
PREDICTION_FRAMES = 15  # frames - predict object position when temporarily lost
OUT_OF_SCOPE_FRAMES = 45  # frames - mark as theft when object leaves camera scope

# Face detection (optimized for PC)
FACE_DETECTION_METHOD = "dnn"  # "dnn" or "haar" - DNN for better accuracy on PC
FACE_CONFIDENCE_THRESHOLD = 0.7  # DNN face detection confidence
FACE_SCALE_FACTOR = 1.1  # Haar cascade scale factor
FACE_MIN_NEIGHBORS = 5  # Haar cascade min neighbors
FACE_DETECTION_INTERVAL = 1  # Detect faces every frame on PC (was 3 for Raspberry Pi)

# Model paths
YOLO_MODEL = "yolov8m.pt"  # YOLOv8 medium model (balanced accuracy and speed with multithreading)
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
DRAW_TRACKS = True  # Draw object tracks

# Object filtering
IGNORE_CLASSES = ["person"]  # Classes to ignore for theft detection
TRACKED_CLASSES = []  # Empty means track all non-person objects, otherwise only track specified classes

# Tracking settings
MAX_POSITION_HISTORY = 100  # Maximum number of positions to keep in object history
THIEF_PROXIMITY_THRESHOLD = 200  # pixels - distance threshold to mark person as thief when object disappears
                                  # ~15% of 1280px width - reasonable proximity for person to be involved
THIEF_STATUS_TIMEOUT = 300  # frames - how long to keep thief status before clearing (10 sec at 30fps)
PERSON_MISSING_TIMEOUT = 90  # frames - remove tracked person if missing this long (3 sec at 30fps)

# Multithreading settings
ENABLE_MULTITHREADING = True  # Enable multithreading for faster processing
FRAME_QUEUE_SIZE = 8  # Size of frame buffer queue
PROCESSING_THREADS = 2  # Number of processing threads (face detection, logging)
CAPTURE_THREAD_PRIORITY = True  # Give capture thread higher priority

# Display colors (BGR format)
COLOR_PERSON = (0, 255, 0)  # Green for normal persons
COLOR_OBJECT = (255, 0, 0)  # Blue for objects
COLOR_THIEF = (0, 0, 255)  # Red for persons marked as thieves
COLOR_FILTERED = (128, 128, 128)  # Gray for filtered objects
