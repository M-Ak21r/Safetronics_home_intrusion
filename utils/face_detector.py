"""
Face detection module supporting both DNN and Haar Cascade methods
"""
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional


class FaceDetector:
    """Face detector using either DNN or Haar Cascade"""
    
    def __init__(self, method: str = "dnn", confidence_threshold: float = 0.7,
                 dnn_proto: Optional[str] = None, dnn_model: Optional[str] = None,
                 haar_cascade: Optional[str] = None):
        """
        Initialize face detector
        
        Args:
            method: "dnn" or "haar"
            confidence_threshold: Minimum confidence for DNN detections
            dnn_proto: Path to DNN prototxt file
            dnn_model: Path to DNN caffemodel file
            haar_cascade: Path to Haar cascade XML file
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.net = None
        self.cascade = None
        
        if method == "dnn":
            self._init_dnn(dnn_proto, dnn_model)
        elif method == "haar":
            self._init_haar(haar_cascade)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'dnn' or 'haar'")
    
    def _init_dnn(self, proto_path: Optional[str], model_path: Optional[str]):
        """Initialize DNN-based face detector with GPU acceleration"""
        try:
            if proto_path and model_path and os.path.exists(proto_path) and os.path.exists(model_path):
                self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                
                # Try to use GPU acceleration for DNN if available
                try:
                    # Check if CUDA is available for OpenCV DNN
                    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                        print(f"✓ DNN face detector loaded with CUDA acceleration")
                    else:
                        print(f"DNN face detector loaded successfully (CPU)")
                except (AttributeError, Exception) as e:
                    # CUDA not available for OpenCV DNN, use CPU
                    print(f"DNN face detector loaded successfully (CPU)")
            else:
                print(f"Warning: DNN model files not found. Face detection will be disabled.")
                print(f"Download from: https://github.com/opencv/opencv_3rdparty/tree/dnn_samples_face_detector_20170830")
                self.net = None
        except Exception as e:
            print(f"Error loading DNN model: {e}")
            self.net = None
    
    def _init_haar(self, cascade_path: Optional[str]):
        """Initialize Haar Cascade face detector"""
        try:
            # Try custom path first
            if cascade_path and os.path.exists(cascade_path):
                self.cascade = cv2.CascadeClassifier(cascade_path)
            else:
                # Try OpenCV's built-in cascade
                cascade_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_file):
                    self.cascade = cv2.CascadeClassifier(cascade_file)
                    print(f"Haar cascade face detector loaded from OpenCV")
                else:
                    print(f"Warning: Haar cascade not found. Face detection will be disabled.")
                    self.cascade = None
            
            # Validate cascade is properly loaded
            if self.cascade is not None:
                # Check if cascade is valid by testing the empty() method if available
                try:
                    if hasattr(self.cascade, 'empty') and self.cascade.empty():
                        self.cascade = None
                    else:
                        print(f"Haar cascade face detector loaded successfully")
                except:
                    # If empty() check fails, assume cascade is valid
                    print(f"Haar cascade face detector loaded successfully")
            
            if self.cascade is None:
                print(f"Warning: Haar cascade could not be initialized properly.")
        except Exception as e:
            print(f"Error loading Haar cascade: {e}")
            self.cascade = None
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of face bounding boxes as (x, y, w, h)
        """
        if self.method == "dnn" and self.net is not None:
            return self._detect_dnn(frame)
        elif self.method == "haar" and self.cascade is not None:
            return self._detect_haar(frame)
        else:
            return []
    
    def _detect_dnn(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using DNN with GPU acceleration"""
        h, w = frame.shape[:2]
        # Create blob with swapRB=False for optimized processing
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), 
                                     (104.0, 177.0, 123.0), swapRB=False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x2, y2 = box.astype(int)
                # Convert to (x, y, w, h) format
                faces.append((x, y, x2 - x, y2 - y))
        
        return faces
    
    def _detect_haar(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return [(x, y, w, h) for (x, y, w, h) in faces]
    
    def crop_face(self, frame: np.ndarray, face_box: Tuple[int, int, int, int], 
                  padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Crop face from frame with optional padding
        
        Args:
            frame: Input frame
            face_box: Face bounding box (x, y, w, h)
            padding: Padding ratio (0.2 = 20% padding)
            
        Returns:
            Cropped face image or None if invalid
        """
        x, y, w, h = face_box
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        return frame[y1:y2, x1:x2]
