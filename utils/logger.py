"""
Logging utilities for theft detection events
"""
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
import cv2


class TheftLogger:
    """Logger for theft detection events"""
    
    def __init__(self, log_dir: str = "logs", evidence_dir: str = "evidence"):
        """
        Initialize logger
        
        Args:
            log_dir: Directory for JSON log files
            evidence_dir: Directory for evidence images
        """
        self.log_dir = log_dir
        self.evidence_dir = evidence_dir
        self.face_dir = os.path.join(evidence_dir, "faces")
        self.frame_dir = os.path.join(evidence_dir, "frames")
        
        # Create directories if they don't exist
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.face_dir, exist_ok=True)
        os.makedirs(self.frame_dir, exist_ok=True)
        
        self.current_log_file = self._get_log_filename()
    
    def _get_log_filename(self) -> str:
        """Generate log filename with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.log_dir, f"theft_log_{timestamp}.json")
    
    def log_theft_event(self, event_type: str, object_info: Dict[str, Any],
                       frame: Optional[Any] = None, faces: Optional[List[Any]] = None) -> str:
        """
        Log a theft event
        
        Args:
            event_type: Type of event ("displacement", "disappearance")
            object_info: Information about the object involved
            frame: Optional frame to save as evidence
            faces: Optional list of detected faces
            
        Returns:
            Event ID (timestamp-based)
        """
        timestamp = datetime.now()
        event_id = timestamp.strftime("%Y%m%d_%H%M%S_%f")
        
        event_data = {
            "event_id": event_id,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "object": {
                "track_id": object_info.get("track_id"),
                "class_name": object_info.get("class_name"),
                "confidence": object_info.get("confidence"),
                "bbox": object_info.get("bbox"),
                "displacement": object_info.get("displacement"),
            },
            "faces_detected": len(faces) if faces else 0,
            "evidence": {}
        }
        
        # Save annotated frame
        if frame is not None:
            frame_filename = f"frame_{event_id}.jpg"
            frame_path = os.path.join(self.frame_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            event_data["evidence"]["frame"] = frame_path
        
        # Save face crops
        if faces:
            face_files = []
            for i, face_img in enumerate(faces):
                face_filename = f"face_{event_id}_{i}.jpg"
                face_path = os.path.join(self.face_dir, face_filename)
                cv2.imwrite(face_path, face_img)
                face_files.append(face_path)
            event_data["evidence"]["faces"] = face_files
        
        # Append to log file
        self._append_to_log(event_data)
        
        return event_id
    
    def _append_to_log(self, event_data: Dict[str, Any]):
        """Append event data to JSON log file"""
        # Read existing log or create new
        if os.path.exists(self.current_log_file):
            with open(self.current_log_file, 'r') as f:
                try:
                    log_data = json.load(f)
                except json.JSONDecodeError:
                    log_data = {"events": []}
        else:
            log_data = {"events": []}
        
        # Append new event
        log_data["events"].append(event_data)
        
        # Write back to file
        with open(self.current_log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def get_recent_events(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent theft events
        
        Args:
            count: Number of recent events to retrieve
            
        Returns:
            List of recent events
        """
        if not os.path.exists(self.current_log_file):
            return []
        
        with open(self.current_log_file, 'r') as f:
            try:
                log_data = json.load(f)
                events = log_data.get("events", [])
                return events[-count:]
            except json.JSONDecodeError:
                return []
