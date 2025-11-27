"""
Core theft detection module using YOLOv8 object tracking and face detection
with multithreading support for faster processing
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import threading
import queue
import os

from utils.geometry import calculate_iou, calculate_center, calculate_distance, xywh_to_xyxy
from utils.face_detector import FaceDetector
from utils.logger import TheftLogger
import config


class TrackedPerson:
    """Represents a tracked person"""
    
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        self.track_id = track_id
        self.confidence = confidence
        self.last_bbox = bbox
        self.last_position = calculate_center(bbox)
        self.positions = [self.last_position]  # Store position history for velocity estimation
        self.is_thief = False  # Marked as thief when nearby object disappears
        self.thief_marked_frame = None  # Frame when marked as thief
        self.missing_frames = 0  # Count missing frames
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.velocity = (0.0, 0.0)  # Estimated velocity for prediction
        self.captured_image = None  # Store captured image when near object
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update person with new detection"""
        old_position = self.last_position
        self.last_bbox = bbox
        self.last_position = calculate_center(bbox)
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.missing_frames = 0
        
        # Update velocity estimation
        self.velocity = (
            self.last_position[0] - old_position[0],
            self.last_position[1] - old_position[1]
        )
        
        # Keep position history
        self.positions.append(self.last_position)
        if len(self.positions) > config.MAX_POSITION_HISTORY:
            self.positions = self.positions[-config.MAX_POSITION_HISTORY:]
    
    def mark_as_thief(self, frame_count: int):
        """Mark person as thief"""
        self.is_thief = True
        self.thief_marked_frame = frame_count
    
    def mark_missing(self):
        """Mark person as missing in current frame"""
        self.missing_frames += 1
    
    def predict_position(self) -> Tuple[float, float]:
        """Predict current position based on velocity"""
        return (
            self.last_position[0] + self.velocity[0] * self.missing_frames,
            self.last_position[1] + self.velocity[1] * self.missing_frames
        )
    
    def should_clear_thief_status(self, current_frame: int) -> bool:
        """Check if thief status should be cleared based on timeout"""
        if not self.is_thief or self.thief_marked_frame is None:
            return False
        from config import THIEF_STATUS_TIMEOUT
        return (current_frame - self.thief_marked_frame) > THIEF_STATUS_TIMEOUT


class TrackedObject:
    """Represents a tracked object with its history and velocity prediction"""
    
    def __init__(self, track_id: int, class_name: str, bbox: Tuple[float, float, float, float],
                 confidence: float):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.positions = [calculate_center(bbox)]
        self.last_bbox = bbox
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.missing_frames = 0
        self.is_filtered = False  # Overlaps with person
        self.total_displacement = 0.0
        self.is_being_stolen = False  # Object is being stolen
        self.velocity = (0.0, 0.0)  # Velocity for prediction during occlusion
        self.predicted_bbox = None  # Predicted bbox when temporarily lost
        self.out_of_scope = False  # Flag for when object leaves camera view
        self.nearby_persons_history = []  # Track persons who were near this object
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update object with new detection"""
        center = calculate_center(bbox)
        if len(self.positions) > 0:
            displacement = calculate_distance(self.positions[-1], center)
            self.total_displacement += displacement
            
            # Update velocity estimation
            old_center = self.positions[-1]
            self.velocity = (center[0] - old_center[0], center[1] - old_center[1])
        
        self.positions.append(center)
        self.last_bbox = bbox
        self.predicted_bbox = None  # Clear prediction when detected
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.missing_frames = 0
        self.out_of_scope = False
        
        # Keep only recent positions for memory efficiency
        from config import MAX_POSITION_HISTORY
        if len(self.positions) > MAX_POSITION_HISTORY:
            self.positions = self.positions[-MAX_POSITION_HISTORY:]
    
    def get_displacement(self) -> float:
        """Calculate total displacement since first detection"""
        if len(self.positions) < 2:
            return 0.0
        return calculate_distance(self.positions[0], self.positions[-1])
    
    def mark_missing(self):
        """Mark object as missing in current frame"""
        self.missing_frames += 1
        
        # Update predicted position
        if self.missing_frames <= config.PREDICTION_FRAMES:
            self._update_predicted_bbox()
    
    def _update_predicted_bbox(self):
        """Predict bbox based on velocity"""
        if self.last_bbox is None:
            return
        
        x1, y1, x2, y2 = self.last_bbox
        vx, vy = self.velocity
        
        # Apply velocity to predict new position
        pred_x1 = x1 + vx * self.missing_frames
        pred_y1 = y1 + vy * self.missing_frames
        pred_x2 = x2 + vx * self.missing_frames
        pred_y2 = y2 + vy * self.missing_frames
        
        self.predicted_bbox = (pred_x1, pred_y1, pred_x2, pred_y2)
    
    def get_current_bbox(self) -> Tuple[float, float, float, float]:
        """Get current or predicted bbox"""
        if self.missing_frames > 0 and self.predicted_bbox is not None:
            return self.predicted_bbox
        return self.last_bbox
    
    def check_out_of_scope(self, frame_width: int, frame_height: int) -> bool:
        """Check if object has moved out of camera scope"""
        bbox = self.get_current_bbox()
        if bbox is None:
            return False
        
        x1, y1, x2, y2 = bbox
        margin = 10  # pixels margin
        
        # Check if predicted position is outside frame boundaries
        if x2 < margin or x1 > frame_width - margin:
            self.out_of_scope = True
            return True
        if y2 < margin or y1 > frame_height - margin:
            self.out_of_scope = True
            return True
        
        return False
    
    def add_nearby_person(self, person_id: int, frame_count: int):
        """Record a person who was near this object"""
        self.nearby_persons_history.append({
            'person_id': person_id,
            'frame': frame_count,
            'timestamp': datetime.now()
        })
        # Keep only recent history
        if len(self.nearby_persons_history) > 50:
            self.nearby_persons_history = self.nearby_persons_history[-50:]


class FrameBuffer:
    """Thread-safe frame buffer for multithreaded processing"""
    
    def __init__(self, maxsize: int = 8):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def put(self, frame: np.ndarray, frame_number: int) -> bool:
        """Add frame to buffer, dropping oldest if full"""
        try:
            self.queue.put_nowait((frame, frame_number))
            with self.lock:
                self.latest_frame = (frame.copy(), frame_number)
            return True
        except queue.Full:
            # Drop oldest frame and add new one
            try:
                self.queue.get_nowait()
                self.queue.put_nowait((frame, frame_number))
                with self.lock:
                    self.latest_frame = (frame.copy(), frame_number)
                return True
            except (queue.Empty, queue.Full):
                return False
    
    def get(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, int]]:
        """Get frame from buffer"""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_latest(self) -> Optional[Tuple[np.ndarray, int]]:
        """Get most recent frame without removing from queue"""
        with self.lock:
            return self.latest_frame
    
    def clear(self):
        """Clear all frames from buffer"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break


class TheftDetector:
    """Main theft detection system with multithreading support"""
    
    def __init__(self, model_path: str = None, video_source: Any = 0):
        """
        Initialize theft detector
        
        Args:
            model_path: Path to YOLOv8 model (default: yolov8m.pt)
            video_source: Video source (camera index or video file path)
        """
        # Load YOLOv8 model
        self.model_path = model_path or config.YOLO_MODEL
        print(f"Loading YOLOv8 model: {self.model_path}")
        self.yolo = YOLO(self.model_path)
        
        # Initialize face detector
        self.face_detector = FaceDetector(
            method=config.FACE_DETECTION_METHOD,
            confidence_threshold=config.FACE_CONFIDENCE_THRESHOLD,
            dnn_proto=config.DNN_PROTO,
            dnn_model=config.DNN_MODEL,
            haar_cascade=config.HAAR_CASCADE
        )
        
        # Initialize logger
        self.logger = TheftLogger(config.LOG_DIR, config.EVIDENCE_DIR)
        
        # Tracking state
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.tracked_persons: Dict[int, TrackedPerson] = {}
        self.person_boxes: List[Tuple[float, float, float, float]] = []
        self.frame_count = 0
        self.frame_width = config.FRAME_WIDTH
        self.frame_height = config.FRAME_HEIGHT
        
        # Video capture
        self.video_source = video_source
        self.cap = None
        
        # Multithreading components
        self.frame_buffer = FrameBuffer(maxsize=config.FRAME_QUEUE_SIZE)
        self.result_buffer = FrameBuffer(maxsize=config.FRAME_QUEUE_SIZE)
        self.capture_thread = None
        self.processing_thread = None
        self.running = False
        self.lock = threading.Lock()
        
        # Thread-safe event queue for theft events
        self.event_queue = queue.Queue()
        
        # Custom tracker config path
        self.tracker_config = self._get_tracker_config()
    
    def _get_tracker_config(self) -> str:
        """Get path to custom tracker config"""
        custom_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "custom_bytetrack.yaml"
        )
        if os.path.exists(custom_config):
            print(f"Using custom tracker config: {custom_config}")
            return custom_config
        print("Using default ByteTrack config")
        return "bytetrack.yaml"
    
    def start_capture(self) -> bool:
        """Start video capture"""
        self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return False
        
        # Set resolution if using camera
        if isinstance(self.video_source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        # Get actual frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video capture started from source: {self.video_source}")
        print(f"Frame dimensions: {self.frame_width}x{self.frame_height}")
        return True
    
    def stop_capture(self):
        """Stop video capture and threads"""
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
    def _capture_frames(self):
        """Thread function for capturing frames"""
        frame_number = 0
        while self.running:
            if self.cap is None or not self.cap.isOpened():
                break
            
            ret, frame = self.cap.read()
            if not ret:
                # End of video
                self.running = False
                break
            
            frame_number += 1
            self.frame_buffer.put(frame, frame_number)
    
    def _process_frames(self):
        """Thread function for processing frames with YOLO"""
        while self.running:
            frame_data = self.frame_buffer.get(timeout=0.1)
            if frame_data is None:
                continue
            
            frame, frame_number = frame_data
            
            # Process frame
            annotated_frame, theft_events = self.process_frame(frame)
            
            # Put result in output buffer
            self.result_buffer.put(annotated_frame, frame_number)
            
            # Add theft events to queue
            for event in theft_events:
                try:
                    self.event_queue.put_nowait(event)
                except queue.Full:
                    pass
    
    def _filter_overlapping_with_persons(self, detections: List[Any]) -> List[bool]:
        """
        Determine which objects overlap with person boxes
        
        Args:
            detections: List of YOLO detections
            
        Returns:
            List of boolean flags indicating if each detection is filtered
        """
        filtered = []
        
        for detection in detections:
            is_filtered = False
            det_box = detection[:4]  # x1, y1, x2, y2
            
            # Check overlap with any person box
            for person_box in self.person_boxes:
                iou = calculate_iou(det_box, person_box)
                if iou > config.IOU_THRESHOLD:
                    is_filtered = True
                    break
            
            filtered.append(is_filtered)
        
        return filtered
    
    def _find_nearby_persons(self, object_box: Tuple[float, float, float, float]) -> List[int]:
        """
        Find tracked persons near an object location
        
        Args:
            object_box: Object bounding box (x1, y1, x2, y2)
            
        Returns:
            List of person track IDs that are nearby
        """
        nearby_person_ids = []
        object_center = calculate_center(object_box)
        
        for person_id, person in self.tracked_persons.items():
            person_center = person.last_position
            distance = calculate_distance(object_center, person_center)
            
            # Check if person is within threshold distance
            if distance <= config.THIEF_PROXIMITY_THRESHOLD:
                nearby_person_ids.append(person_id)
        
        return nearby_person_ids
    
    def _detect_faces_nearby(self, frame: np.ndarray, 
                            object_box: Tuple[float, float, float, float],
                            search_radius: int = 200) -> List[np.ndarray]:
        """
        Detect faces near an object location
        
        Args:
            frame: Input frame
            object_box: Object bounding box (x1, y1, x2, y2)
            search_radius: Pixel radius to search for faces
            
        Returns:
            List of cropped face images
        """
        # Detect faces in frame
        face_boxes = self.face_detector.detect_faces(frame)
        
        nearby_faces = []
        object_center = calculate_center(object_box)
        
        for face_box in face_boxes:
            # Convert face_box from (x, y, w, h) to (x1, y1, x2, y2)
            face_xyxy = xywh_to_xyxy(face_box)
            face_center = calculate_center(face_xyxy)
            
            # Check if face is within search radius
            distance = calculate_distance(object_center, face_center)
            if distance <= search_radius:
                face_crop = self.face_detector.crop_face(frame, face_box)
                if face_crop is not None:
                    nearby_faces.append(face_crop)
        
        return nearby_faces
    
    def _check_theft_conditions(self, obj: TrackedObject, frame: np.ndarray) -> Optional[str]:
        """
        Check if theft conditions are met for an object
        
        Args:
            obj: Tracked object
            frame: Current frame
            
        Returns:
            Event type if theft detected, None otherwise
        """
        # Skip filtered objects (overlapping with persons)
        if obj.is_filtered:
            return None
        
        # Check for large displacement
        displacement = obj.get_displacement()
        if displacement > config.DISPLACEMENT_THRESHOLD:
            return "displacement"
        
        # Check for out-of-scope (object left camera view)
        if obj.missing_frames >= config.OUT_OF_SCOPE_FRAMES:
            if obj.check_out_of_scope(self.frame_width, self.frame_height):
                return "out_of_scope"
        
        # Check for disappearance
        if obj.missing_frames >= config.DISAPPEARANCE_FRAMES:
            return "disappearance"
        
        return None
    
    def _capture_person_near_object(self, frame: np.ndarray, person: TrackedPerson,
                                    obj: TrackedObject) -> Optional[np.ndarray]:
        """
        Capture image of person who is near a tracked object
        
        Args:
            frame: Current frame
            person: TrackedPerson near the object
            obj: TrackedObject being monitored
            
        Returns:
            Cropped image of person or None
        """
        if person.last_bbox is None:
            return None
        
        x1, y1, x2, y2 = map(int, person.last_bbox)
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2].copy()
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame for theft detection
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Tuple of (annotated frame, list of theft events)
        """
        self.frame_count += 1
        annotated_frame = frame.copy()
        theft_events = []
        
        # Run YOLOv8 tracking with optimized parameters and custom tracker
        results = self.yolo.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_IOU_THRESHOLD,
            verbose=False,
            tracker=self.tracker_config
        )
        
        if results and len(results) > 0:
            result = results[0]
            
            # Process all detections
            current_object_tracks = set()
            current_person_tracks = set()
            
            if result.boxes is not None and result.boxes.id is not None:
                # Get detection data
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                # First pass: collect person boxes for filtering
                self.person_boxes = []
                for i in range(len(boxes)):
                    cls_id = class_ids[i]
                    class_name = self.yolo.names[cls_id]
                    if class_name == "person":
                        self.person_boxes.append(tuple(boxes[i]))
                
                # Filter detections overlapping with persons
                filtered_flags = self._filter_overlapping_with_persons(boxes)
                
                # Second pass: update persons and track who is near objects
                for i in range(len(boxes)):
                    cls_id = class_ids[i]
                    class_name = self.yolo.names[cls_id]
                    track_id = track_ids[i]
                    bbox = tuple(boxes[i])
                    conf = float(confidences[i])
                    
                    if class_name == "person":
                        current_person_tracks.add(track_id)
                        
                        # Update or create tracked person
                        if track_id in self.tracked_persons:
                            person = self.tracked_persons[track_id]
                            person.update(bbox, conf)
                        else:
                            person = TrackedPerson(track_id, bbox, conf)
                            self.tracked_persons[track_id] = person
                        
                        # Capture person image if near any tracked object
                        person_center = person.last_position
                        for obj_id, obj in self.tracked_objects.items():
                            obj_center = calculate_center(obj.get_current_bbox())
                            distance = calculate_distance(person_center, obj_center)
                            if distance <= config.THIEF_PROXIMITY_THRESHOLD:
                                # Record this person was near the object
                                obj.add_nearby_person(track_id, self.frame_count)
                                # Capture person image
                                if person.captured_image is None:
                                    person.captured_image = self._capture_person_near_object(
                                        frame, person, obj
                                    )
                        
                        # Clear thief status if timeout reached
                        if person.should_clear_thief_status(self.frame_count):
                            person.is_thief = False
                            person.thief_marked_frame = None
                            print(f"ℹ️  Person #{track_id} thief status cleared (timeout)")
                        
                        # Draw person bounding box
                        if config.DRAW_BOXES:
                            x1, y1, x2, y2 = map(int, bbox)
                            # Red if thief, green if normal person
                            color = config.COLOR_THIEF if person.is_thief else config.COLOR_PERSON
                            thickness = 3 if person.is_thief else 2
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            label = f"Person #{track_id}"
                            if person.is_thief:
                                label += " [THIEF]"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Third pass: update objects
                for i in range(len(boxes)):
                    cls_id = class_ids[i]
                    class_name = self.yolo.names[cls_id]
                    track_id = track_ids[i]
                    bbox = tuple(boxes[i])
                    conf = float(confidences[i])
                    
                    # Skip person class
                    if class_name in config.IGNORE_CLASSES:
                        continue
                    
                    # Skip if tracked classes is specified and this class is not in it
                    if config.TRACKED_CLASSES and class_name not in config.TRACKED_CLASSES:
                        continue
                    
                    current_object_tracks.add(track_id)
                    
                    # Update or create tracked object
                    if track_id in self.tracked_objects:
                        obj = self.tracked_objects[track_id]
                        obj.update(bbox, conf)
                        obj.is_filtered = filtered_flags[i]
                    else:
                        obj = TrackedObject(track_id, class_name, bbox, conf)
                        obj.is_filtered = filtered_flags[i]
                        self.tracked_objects[track_id] = obj
                    
                    # Check if object is being stolen (overlapping with person)
                    obj.is_being_stolen = obj.is_filtered
                    
                    # Draw object bounding box
                    if config.DRAW_BOXES:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Use blue for objects, gray for filtered (held by person)
                        if obj.is_being_stolen:
                            color = config.COLOR_FILTERED
                        else:
                            color = config.COLOR_OBJECT
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name} #{track_id}"
                        if obj.is_filtered:
                            label += " [Held]"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw track
                    if config.DRAW_TRACKS and len(obj.positions) > 1:
                        points = [(int(x), int(y)) for x, y in obj.positions[-20:]]
                        for j in range(len(points) - 1):
                            cv2.line(annotated_frame, points[j], points[j+1], color, 1)
            
            # Mark missing objects and persons, and draw predicted boxes
            for track_id in list(self.tracked_objects.keys()):
                if track_id not in current_object_tracks:
                    obj = self.tracked_objects[track_id]
                    obj.mark_missing()
                    
                    # Draw predicted box for temporarily lost objects
                    if config.DRAW_BOXES and obj.missing_frames <= config.PREDICTION_FRAMES:
                        pred_bbox = obj.get_current_bbox()
                        if pred_bbox is not None:
                            x1, y1, x2, y2 = map(int, pred_bbox)
                            # Draw dashed box for predicted position (orange color)
                            color = (0, 165, 255)  # Orange for predicted
                            # Draw dashed rectangle
                            for j in range(0, int(x2-x1), 10):
                                cv2.line(annotated_frame, (x1+j, y1), (min(x1+j+5, x2), y1), color, 2)
                                cv2.line(annotated_frame, (x1+j, y2), (min(x1+j+5, x2), y2), color, 2)
                            for j in range(0, int(y2-y1), 10):
                                cv2.line(annotated_frame, (x1, y1+j), (x1, min(y1+j+5, y2)), color, 2)
                                cv2.line(annotated_frame, (x2, y1+j), (x2, min(y1+j+5, y2)), color, 2)
                            
                            label = f"{obj.class_name} #{track_id} [Predicted]"
                            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            for track_id in list(self.tracked_persons.keys()):
                if track_id not in current_person_tracks:
                    self.tracked_persons[track_id].mark_missing()
            
            # Check for theft events (separate from deletion loop)
            objects_to_remove = []
            for track_id, obj in self.tracked_objects.items():
                event_type = self._check_theft_conditions(obj, frame)
                
                if event_type:
                    # Find nearby persons from history and current proximity
                    nearby_persons = self._find_nearby_persons(obj.get_current_bbox())
                    
                    # Also check historical nearby persons
                    historical_persons = set(
                        entry['person_id'] for entry in obj.nearby_persons_history
                        if self.frame_count - entry['frame'] < config.DISAPPEARANCE_FRAMES
                    )
                    all_nearby_persons = list(set(nearby_persons) | historical_persons)
                    
                    if event_type in ("disappearance", "out_of_scope") and all_nearby_persons:
                        # Mark nearby persons as thieves
                        for person_id in all_nearby_persons:
                            if person_id in self.tracked_persons:
                                self.tracked_persons[person_id].mark_as_thief(self.frame_count)
                                print(f"🚨 Person #{person_id} marked as THIEF - {obj.class_name} #{track_id} {event_type}")
                    
                    # Detect nearby faces (every N frames for performance)
                    nearby_faces = []
                    if self.frame_count % config.FACE_DETECTION_INTERVAL == 0:
                        nearby_faces = self._detect_faces_nearby(frame, obj.get_current_bbox())
                    
                    # Collect person captures for evidence
                    person_captures = []
                    for person_id in all_nearby_persons:
                        if person_id in self.tracked_persons:
                            person = self.tracked_persons[person_id]
                            if person.captured_image is not None:
                                person_captures.append(person.captured_image)
                    
                    # Log theft event if face detected, person marked as thief, or out of scope
                    if nearby_faces or all_nearby_persons or event_type == "out_of_scope":
                        object_info = {
                            "track_id": obj.track_id,
                            "class_name": obj.class_name,
                            "confidence": obj.confidence,
                            "bbox": obj.get_current_bbox(),
                            "displacement": obj.get_displacement(),
                            "nearby_persons": all_nearby_persons,
                            "out_of_scope": obj.out_of_scope
                        }
                        
                        # Combine faces and person captures for evidence
                        all_evidence_images = nearby_faces + person_captures
                        
                        event_id = self.logger.log_theft_event(
                            event_type,
                            object_info,
                            annotated_frame,
                            all_evidence_images
                        )
                        
                        theft_events.append({
                            "event_id": event_id,
                            "event_type": event_type,
                            "object": object_info,
                            "faces_count": len(nearby_faces),
                            "person_captures": len(person_captures),
                            "thief_persons": all_nearby_persons
                        })
                        
                        print(f"⚠️  THEFT DETECTED: {event_type} - {obj.class_name} #{obj.track_id} "
                              f"(displacement: {obj.get_displacement():.1f}px, faces: {len(nearby_faces)}, "
                              f"nearby persons: {len(all_nearby_persons)})")
                        
                        # Draw alert on frame
                        alert_text = f"THEFT: {event_type.upper()}!"
                        if all_nearby_persons:
                            alert_text += f" - Person(s) {all_nearby_persons}"
                        cv2.putText(annotated_frame, alert_text,
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Mark object for removal if it's been missing too long
                    if event_type in ("disappearance", "out_of_scope"):
                        objects_to_remove.append(track_id)
            
            # Remove objects that have disappeared (done after iteration)
            for track_id in objects_to_remove:
                if track_id in self.tracked_objects:
                    del self.tracked_objects[track_id]
            
            # Clean up persons that have been missing too long
            persons_to_remove = []
            for person_id, person in self.tracked_persons.items():
                if person.missing_frames > config.PERSON_MISSING_TIMEOUT:
                    persons_to_remove.append(person_id)
            
            for person_id in persons_to_remove:
                if person_id in self.tracked_persons:
                    del self.tracked_persons[person_id]
        
        # Draw status info
        status_text = f"Frame: {self.frame_count} | Objects: {len(self.tracked_objects)} | " \
                     f"Persons: {len(self.tracked_persons)}"
        cv2.putText(annotated_frame, status_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame, theft_events
    
    def run(self):
        """Run the theft detection system with optional multithreading"""
        if not self.start_capture():
            return
        
        print("="*60)
        print("Theft Detection System Started")
        print(f"Multithreading: {'Enabled' if config.ENABLE_MULTITHREADING else 'Disabled'}")
        print("="*60)
        print("Press 'q' to quit")
        print()
        
        if config.ENABLE_MULTITHREADING:
            self._run_multithreaded()
        else:
            self._run_single_threaded()
    
    def _run_single_threaded(self):
        """Run in single-threaded mode (original behavior)"""
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                annotated_frame, theft_events = self.process_frame(frame)
                
                # Display frame
                if config.SHOW_DISPLAY:
                    cv2.imshow("Theft Detection", annotated_frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping theft detection...")
                    break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.stop_capture()
            print("Theft detection stopped")
    
    def _run_multithreaded(self):
        """Run with multithreading for better performance"""
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread  
        self.processing_thread = threading.Thread(target=self._process_frames, daemon=True)
        self.processing_thread.start()
        
        print("Capture and processing threads started")
        
        try:
            last_frame_number = 0
            while self.running:
                # Get processed result
                result = self.result_buffer.get(timeout=0.1)
                
                if result is not None:
                    annotated_frame, frame_number = result
                    last_frame_number = frame_number
                    
                    # Display frame
                    if config.SHOW_DISPLAY:
                        cv2.imshow("Theft Detection", annotated_frame)
                
                # Process any theft events in queue
                while not self.event_queue.empty():
                    try:
                        event = self.event_queue.get_nowait()
                        # Events are already logged in process_frame
                    except queue.Empty:
                        break
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping theft detection...")
                    self.running = False
                    break
                
                # Check if capture thread has ended (end of video)
                if not self.capture_thread.is_alive() and self.frame_buffer.queue.empty():
                    # Wait for processing to complete
                    if self.result_buffer.queue.empty():
                        print("End of video stream")
                        self.running = False
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.running = False
        
        finally:
            self.stop_capture()
            print("Theft detection stopped")
