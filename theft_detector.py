"""
Core theft detection module using YOLOv8 object tracking and hand-based theft detection.
Detects stationary objects, creates hitboxes around them, and triggers theft alerts
when a person's hand enters the hitbox.
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

from utils.geometry import calculate_iou, calculate_center, calculate_distance, xywh_to_xyxy, is_point_in_box
from utils.face_detector import FaceDetector
from utils.logger import TheftLogger
import config


class TrackedPerson:
    """Represents a tracked person with hand detection for theft attempts"""
    
    def __init__(self, track_id: int, bbox: Tuple[float, float, float, float], confidence: float):
        self.track_id = track_id
        self.confidence = confidence
        self.last_bbox = bbox
        self.last_position = calculate_center(bbox)
        self.is_thief = False  # Marked as thief when hand enters hitbox
        self.thief_marked_frame = None  # Frame when marked as thief
        self.missing_frames = 0  # Count missing frames
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.captured_image = None  # Store captured image when near object
        self.hand_regions = []  # Store estimated hand regions (left and right)
        self.theft_attempts = 0  # Count theft attempts
        
        # Estimate hand regions immediately
        self._estimate_hand_regions()
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update person with new detection"""
        self.last_bbox = bbox
        self.last_position = calculate_center(bbox)
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.missing_frames = 0
        
        # Estimate hand regions based on person bounding box
        self._estimate_hand_regions()
    
    def _estimate_hand_regions(self):
        """Estimate hand regions from person bounding box"""
        if self.last_bbox is None:
            self.hand_regions = []
            return
        
        x1, y1, x2, y2 = self.last_bbox
        width = x2 - x1
        height = y2 - y1
        
        # Hands are typically at sides, around 60-80% down from top of person box
        # and extend outside the body width
        hand_y_start = y1 + height * 0.4  # Start from 40% down
        hand_y_end = y1 + height * 0.75   # End at 75% down
        
        hand_width = width * 0.3  # Hand region width
        
        # Left hand region (extends to left of person)
        left_hand = (
            x1 - hand_width,          # x1
            hand_y_start,             # y1
            x1 + width * 0.2,         # x2
            hand_y_end                # y2
        )
        
        # Right hand region (extends to right of person)
        right_hand = (
            x2 - width * 0.2,         # x1
            hand_y_start,             # y1
            x2 + hand_width,          # x2
            hand_y_end                # y2
        )
        
        self.hand_regions = [left_hand, right_hand]
    
    def check_hand_in_hitbox(self, hitbox: Tuple[float, float, float, float]) -> bool:
        """Check if any hand region overlaps with an object hitbox"""
        for hand_region in self.hand_regions:
            iou = calculate_iou(hand_region, hitbox)
            if iou > 0:  # Any overlap counts
                return True
        return False
    
    def mark_as_thief(self, frame_count: int):
        """Mark person as thief"""
        self.is_thief = True
        self.thief_marked_frame = frame_count
        self.theft_attempts += 1
    
    def mark_missing(self):
        """Mark person as missing in current frame"""
        self.missing_frames += 1
    
    def should_clear_thief_status(self, current_frame: int) -> bool:
        """Check if thief status should be cleared based on timeout"""
        if not self.is_thief or self.thief_marked_frame is None:
            return False
        from config import THIEF_STATUS_TIMEOUT
        return (current_frame - self.thief_marked_frame) > THIEF_STATUS_TIMEOUT


class StationaryObject:
    """Represents a stationary object being monitored with a hitbox for theft detection"""
    
    def __init__(self, track_id: int, class_name: str, bbox: Tuple[float, float, float, float],
                 confidence: float):
        self.track_id = track_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.hitbox = self._create_hitbox(bbox)  # Expanded hitbox for detection
        self.initial_position = calculate_center(bbox)
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.missing_frames = 0
        self.is_stationary = False  # Becomes true after object is stable
        self.position_history = [calculate_center(bbox)]
        self.stability_frames = 0  # Count frames object has been stable
        self.theft_attempts = []  # Record of theft attempts (person_id, frame, time)
        self.is_being_touched = False  # Hand is currently in hitbox
    
    def _create_hitbox(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """Create expanded hitbox around object for hand detection"""
        x1, y1, x2, y2 = bbox
        
        # Expand hitbox by configured margin
        margin = config.HITBOX_MARGIN
        return (
            x1 - margin,
            y1 - margin,
            x2 + margin,
            y2 + margin
        )
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update object with new detection"""
        old_center = calculate_center(self.bbox)
        new_center = calculate_center(bbox)
        
        # Calculate displacement
        displacement = calculate_distance(old_center, new_center)
        
        # Update position history
        self.position_history.append(new_center)
        if len(self.position_history) > config.MAX_POSITION_HISTORY:
            self.position_history = self.position_history[-config.MAX_POSITION_HISTORY:]
        
        # Check if object is stationary (small movement)
        if displacement < config.STATIONARY_THRESHOLD:
            self.stability_frames += 1
        else:
            self.stability_frames = 0
            self.is_stationary = False
        
        # Mark as stationary after enough stable frames
        if self.stability_frames >= config.STATIONARY_FRAMES:
            self.is_stationary = True
        
        self.bbox = bbox
        self.hitbox = self._create_hitbox(bbox)
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.missing_frames = 0
    
    def mark_missing(self):
        """Mark object as missing in current frame"""
        self.missing_frames += 1
    
    def record_theft_attempt(self, person_id: int, frame_count: int):
        """Record a theft attempt by a person"""
        self.theft_attempts.append({
            'person_id': person_id,
            'frame': frame_count,
            'timestamp': datetime.now()
        })
        # Keep only recent attempts
        if len(self.theft_attempts) > 50:
            self.theft_attempts = self.theft_attempts[-50:]
    
    def get_recent_theft_attempts(self, within_frames: int = 30) -> List[Dict]:
        """Get recent theft attempts within frame window"""
        if not self.theft_attempts:
            return []
        
        current_time = datetime.now()
        recent = []
        for attempt in self.theft_attempts:
            # Consider attempts within last few seconds
            time_diff = (current_time - attempt['timestamp']).total_seconds()
            if time_diff < 5:  # Within 5 seconds
                recent.append(attempt)
        return recent


class FrameBuffer:
    """Thread-safe frame buffer for multithreaded processing"""
    
    def __init__(self, maxsize: int = 8):
        self.queue = queue.Queue(maxsize=maxsize)
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def put(self, frame: np.ndarray, frame_number: int) -> bool:
        """Add frame to buffer, dropping oldest if full"""
        # Copy frame outside lock to avoid holding lock during expensive operation
        frame_copy = frame.copy()
        
        try:
            self.queue.put_nowait((frame, frame_number))
            with self.lock:
                self.latest_frame = (frame_copy, frame_number)
            return True
        except queue.Full:
            # Drop oldest frame and add new one
            # Use a single lock to prevent race conditions
            with self.lock:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass  # Already empty, continue
                
                try:
                    self.queue.put_nowait((frame, frame_number))
                    self.latest_frame = (frame_copy, frame_number)
                    return True
                except queue.Full:
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
            if self.latest_frame is not None:
                return self.latest_frame
            return None
    
    def clear(self):
        """Clear all frames from buffer"""
        with self.lock:
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
    
    def is_empty(self) -> bool:
        """Thread-safe check if buffer is empty"""
        with self.lock:
            return self.queue.empty()


class TheftDetector:
    """Main theft detection system using hand-in-hitbox detection"""
    
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
        
        # Tracking state - use StationaryObject instead of TrackedObject
        self.stationary_objects: Dict[int, StationaryObject] = {}
        self.tracked_persons: Dict[int, TrackedPerson] = {}
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
    
    def _check_hand_in_hitbox(self, person: TrackedPerson, obj: StationaryObject) -> bool:
        """
        Check if person's hand is inside object's hitbox
        
        Args:
            person: TrackedPerson to check
            obj: StationaryObject with hitbox
            
        Returns:
            True if hand is in hitbox
        """
        return person.check_hand_in_hitbox(obj.hitbox)
    
    def _capture_person_image(self, frame: np.ndarray, person: TrackedPerson) -> Optional[np.ndarray]:
        """
        Capture image of person
        
        Args:
            frame: Current frame
            person: TrackedPerson to capture
            
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
        Process a single frame for theft detection using hand-in-hitbox approach
        
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
                
                # First pass: update persons
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
                            
                            # Draw hand regions if configured
                            if config.DRAW_HAND_REGIONS:
                                for hand_region in person.hand_regions:
                                    hx1, hy1, hx2, hy2 = map(int, hand_region)
                                    cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), 
                                                (255, 0, 255), 1)  # Magenta for hand regions
                
                # Second pass: update stationary objects
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
                    
                    # Update or create stationary object
                    if track_id in self.stationary_objects:
                        obj = self.stationary_objects[track_id]
                        obj.update(bbox, conf)
                    else:
                        obj = StationaryObject(track_id, class_name, bbox, conf)
                        self.stationary_objects[track_id] = obj
                    
                    # Draw object bounding box
                    if config.DRAW_BOXES:
                        x1, y1, x2, y2 = map(int, bbox)
                        # Blue for objects, cyan for stationary objects
                        if obj.is_stationary:
                            color = config.COLOR_STATIONARY
                        else:
                            color = config.COLOR_OBJECT
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name} #{track_id}"
                        if obj.is_stationary:
                            label += " [Stationary]"
                        if obj.is_being_touched:
                            label += " [TOUCHED!]"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw hitbox for stationary objects
                    if config.DRAW_HITBOXES and obj.is_stationary:
                        hx1, hy1, hx2, hy2 = map(int, obj.hitbox)
                        # Draw hitbox with dashed line (yellow)
                        color = config.COLOR_HITBOX
                        thickness = 2
                        dash_length = 10
                        
                        # Draw dashed rectangle
                        for j in range(int(hx1), int(hx2), dash_length * 2):
                            cv2.line(annotated_frame, (j, hy1), (min(j + dash_length, hx2), hy1), color, thickness)
                            cv2.line(annotated_frame, (j, hy2), (min(j + dash_length, hx2), hy2), color, thickness)
                        for j in range(int(hy1), int(hy2), dash_length * 2):
                            cv2.line(annotated_frame, (hx1, j), (hx1, min(j + dash_length, hy2)), color, thickness)
                            cv2.line(annotated_frame, (hx2, j), (hx2, min(j + dash_length, hy2)), color, thickness)
            
            # Mark missing objects and persons
            for track_id in list(self.stationary_objects.keys()):
                if track_id not in current_object_tracks:
                    self.stationary_objects[track_id].mark_missing()
            
            for track_id in list(self.tracked_persons.keys()):
                if track_id not in current_person_tracks:
                    self.tracked_persons[track_id].mark_missing()
            
            # Check for theft attempts (hand in hitbox)
            for obj_id, obj in self.stationary_objects.items():
                # Only check stationary objects
                if not obj.is_stationary:
                    continue
                
                obj.is_being_touched = False
                
                # Check each person's hands against this object's hitbox
                for person_id, person in self.tracked_persons.items():
                    if self._check_hand_in_hitbox(person, obj):
                        obj.is_being_touched = True
                        
                        # Record theft attempt
                        obj.record_theft_attempt(person_id, self.frame_count)
                        
                        # Mark person as thief
                        if not person.is_thief:
                            person.mark_as_thief(self.frame_count)
                            print(f"🚨 THEFT ATTEMPT: Person #{person_id} hand entered hitbox of {obj.class_name} #{obj_id}!")
                            
                            # Capture person image as evidence
                            if person.captured_image is None:
                                person.captured_image = self._capture_person_image(frame, person)
                            
                            # Detect face for evidence (throttled for performance)
                            nearby_faces = []
                            if self.frame_count % config.FACE_DETECTION_INTERVAL == 0:
                                nearby_faces = self._detect_faces_nearby(frame, obj.bbox)
                            
                            # Log theft event
                            object_info = {
                                "track_id": obj.track_id,
                                "class_name": obj.class_name,
                                "confidence": obj.confidence,
                                "bbox": obj.bbox,
                                "hitbox": obj.hitbox,
                                "person_id": person_id
                            }
                            
                            evidence_images = nearby_faces
                            if person.captured_image is not None:
                                evidence_images.append(person.captured_image)
                            
                            event_id = self.logger.log_theft_event(
                                "hand_in_hitbox",
                                object_info,
                                annotated_frame,
                                evidence_images
                            )
                            
                            theft_events.append({
                                "event_id": event_id,
                                "event_type": "hand_in_hitbox",
                                "object": object_info,
                                "person_id": person_id,
                                "faces_count": len(nearby_faces)
                            })
                            
                            # Draw alert on frame
                            cv2.putText(annotated_frame, f"THEFT ATTEMPT! Person #{person_id}",
                                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Clean up objects that have been missing too long
            objects_to_remove = []
            for track_id, obj in self.stationary_objects.items():
                if obj.missing_frames > config.OBJECT_MISSING_TIMEOUT:
                    objects_to_remove.append(track_id)
            
            for track_id in objects_to_remove:
                del self.stationary_objects[track_id]
            
            # Clean up persons that have been missing too long
            persons_to_remove = []
            for person_id, person in self.tracked_persons.items():
                if person.missing_frames > config.PERSON_MISSING_TIMEOUT:
                    persons_to_remove.append(person_id)
            
            for person_id in persons_to_remove:
                del self.tracked_persons[person_id]
        
        # Draw status info
        stationary_count = sum(1 for obj in self.stationary_objects.values() if obj.is_stationary)
        status_text = f"Frame: {self.frame_count} | Objects: {len(self.stationary_objects)} ({stationary_count} stationary) | " \
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
                
                # Process any theft events in queue (non-blocking)
                try:
                    while True:
                        event = self.event_queue.get_nowait()
                        # Events are already logged in process_frame
                except queue.Empty:
                    pass  # No more events to process
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopping theft detection...")
                    self.running = False
                    break
                
                # Check if capture thread has ended (end of video)
                if not self.capture_thread.is_alive() and self.frame_buffer.is_empty():
                    # Wait for processing to complete
                    if self.result_buffer.is_empty():
                        print("End of video stream")
                        self.running = False
                        break
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.running = False
        
        finally:
            self.stop_capture()
            print("Theft detection stopped")
