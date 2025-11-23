"""
Core theft detection module using YOLOv8 object tracking and face detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime

from utils.geometry import calculate_iou, calculate_center, calculate_distance, xywh_to_xyxy
from utils.face_detector import FaceDetector
from utils.logger import TheftLogger
import config


class TrackedObject:
    """Represents a tracked object with its history"""
    
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
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update object with new detection"""
        center = calculate_center(bbox)
        if len(self.positions) > 0:
            displacement = calculate_distance(self.positions[-1], center)
            self.total_displacement += displacement
        
        self.positions.append(center)
        self.last_bbox = bbox
        self.confidence = confidence
        self.last_seen = datetime.now()
        self.missing_frames = 0
        
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


class TheftDetector:
    """Main theft detection system"""
    
    def __init__(self, model_path: str = None, video_source: Any = 0):
        """
        Initialize theft detector
        
        Args:
            model_path: Path to YOLOv8 model (default: yolov8n.pt)
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
        self.person_boxes: List[Tuple[float, float, float, float]] = []
        self.frame_count = 0
        
        # Video capture
        self.video_source = video_source
        self.cap = None
    
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
        
        print(f"Video capture started from source: {self.video_source}")
        return True
    
    def stop_capture(self):
        """Stop video capture"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
    
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
        
        # Check for disappearance
        if obj.missing_frames >= config.DISAPPEARANCE_FRAMES:
            return "disappearance"
        
        return None
    
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
        
        # Run YOLOv8 tracking
        results = self.yolo.track(
            frame,
            persist=True,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False
        )
        
        if results and len(results) > 0:
            result = results[0]
            
            # Extract person boxes first
            self.person_boxes = []
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0])
                    class_name = self.yolo.names[cls_id]
                    
                    if class_name == "person":
                        xyxy = box.xyxy[0].cpu().numpy()
                        self.person_boxes.append(tuple(xyxy))
            
            # Process all detections
            current_tracks = set()
            
            if result.boxes is not None and result.boxes.id is not None:
                # Get detection data
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                # Filter detections overlapping with persons
                detections_data = [(boxes[i], confidences[i], class_ids[i], track_ids[i]) 
                                  for i in range(len(boxes))]
                filtered_flags = self._filter_overlapping_with_persons(boxes)
                
                # Update tracked objects
                for i, (box, conf, cls_id, track_id) in enumerate(detections_data):
                    class_name = self.yolo.names[cls_id]
                    
                    # Skip person class
                    if class_name in config.IGNORE_CLASSES:
                        continue
                    
                    # Skip if tracked classes is specified and this class is not in it
                    if config.TRACKED_CLASSES and class_name not in config.TRACKED_CLASSES:
                        continue
                    
                    current_tracks.add(track_id)
                    bbox = tuple(box)
                    
                    # Update or create tracked object
                    if track_id in self.tracked_objects:
                        obj = self.tracked_objects[track_id]
                        obj.update(bbox, float(conf))
                        obj.is_filtered = filtered_flags[i]
                    else:
                        obj = TrackedObject(track_id, class_name, bbox, float(conf))
                        obj.is_filtered = filtered_flags[i]
                        self.tracked_objects[track_id] = obj
                    
                    # Draw bounding box
                    if config.DRAW_BOXES:
                        x1, y1, x2, y2 = map(int, bbox)
                        color = (0, 255, 0) if not obj.is_filtered else (128, 128, 128)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        label = f"{class_name} #{track_id}"
                        if obj.is_filtered:
                            label += " [F]"
                        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw track
                    if config.DRAW_TRACKS and len(obj.positions) > 1:
                        points = [(int(x), int(y)) for x, y in obj.positions[-20:]]
                        for j in range(len(points) - 1):
                            cv2.line(annotated_frame, points[j], points[j+1], (255, 0, 0), 2)
            
            # Mark missing objects
            for track_id in self.tracked_objects.keys():
                if track_id not in current_tracks:
                    self.tracked_objects[track_id].mark_missing()
            
            # Check for theft events (separate from deletion loop)
            objects_to_remove = []
            for track_id, obj in self.tracked_objects.items():
                event_type = self._check_theft_conditions(obj, frame)
                
                if event_type:
                    # Detect nearby faces (every N frames for performance)
                    nearby_faces = []
                    if self.frame_count % config.FACE_DETECTION_INTERVAL == 0:
                        nearby_faces = self._detect_faces_nearby(frame, obj.last_bbox)
                    
                    # Log theft event only if face detected nearby
                    if nearby_faces:
                        object_info = {
                            "track_id": obj.track_id,
                            "class_name": obj.class_name,
                            "confidence": obj.confidence,
                            "bbox": obj.last_bbox,
                            "displacement": obj.get_displacement()
                        }
                        
                        event_id = self.logger.log_theft_event(
                            event_type,
                            object_info,
                            annotated_frame,
                            nearby_faces
                        )
                        
                        theft_events.append({
                            "event_id": event_id,
                            "event_type": event_type,
                            "object": object_info,
                            "faces_count": len(nearby_faces)
                        })
                        
                        print(f"⚠️  THEFT DETECTED: {event_type} - {obj.class_name} #{obj.track_id} "
                              f"(displacement: {obj.get_displacement():.1f}px, faces: {len(nearby_faces)})")
                        
                        # Draw alert on frame
                        cv2.putText(annotated_frame, f"THEFT: {event_type.upper()}!",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    
                    # Mark object for removal if it's been missing too long
                    if event_type == "disappearance":
                        objects_to_remove.append(track_id)
            
            # Remove objects that have disappeared (done after iteration)
            for track_id in objects_to_remove:
                if track_id in self.tracked_objects:
                    del self.tracked_objects[track_id]
        
        # Draw status info
        status_text = f"Frame: {self.frame_count} | Objects: {len(self.tracked_objects)} | " \
                     f"Persons: {len(self.person_boxes)}"
        cv2.putText(annotated_frame, status_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame, theft_events
    
    def run(self):
        """Run the theft detection system"""
        if not self.start_capture():
            return
        
        print("="*60)
        print("Theft Detection System Started")
        print("="*60)
        print("Press 'q' to quit")
        print()
        
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
