#!/usr/bin/env python3
"""
Test script to verify the theft detection system components
"""
import sys
import numpy as np
import cv2

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    try:
        import config
        print("  ✓ config imported")
        
        from utils.geometry import calculate_iou, calculate_center, calculate_distance
        print("  ✓ utils.geometry imported")
        
        from utils.face_detector import FaceDetector
        print("  ✓ utils.face_detector imported")
        
        from utils.logger import TheftLogger
        print("  ✓ utils.logger imported")
        
        from theft_detector import TheftDetector, TrackedObject
        print("  ✓ theft_detector imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_geometry():
    """Test geometry utility functions"""
    print("\nTesting geometry functions...")
    try:
        from utils.geometry import calculate_iou, calculate_center, calculate_distance
        from utils.geometry import xywh_to_xyxy, xyxy_to_xywh
        
        # Test IoU
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 150, 150)
        iou = calculate_iou(box1, box2)
        assert 0 <= iou <= 1, "IoU should be between 0 and 1"
        print(f"  ✓ IoU calculation: {iou:.3f}")
        
        # Test center calculation
        center = calculate_center(box1)
        assert center == (50, 50), "Center calculation incorrect"
        print(f"  ✓ Center calculation: {center}")
        
        # Test distance
        dist = calculate_distance((0, 0), (3, 4))
        assert abs(dist - 5.0) < 0.01, "Distance calculation incorrect"
        print(f"  ✓ Distance calculation: {dist:.1f}")
        
        # Test box format conversion
        xywh = (10, 20, 30, 40)
        xyxy = xywh_to_xyxy(xywh)
        assert xyxy == (10, 20, 40, 60), "xywh_to_xyxy conversion incorrect"
        print(f"  ✓ Box format conversion")
        
        return True
    except Exception as e:
        print(f"  ✗ Geometry test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_face_detector():
    """Test face detector initialization"""
    print("\nTesting face detector...")
    try:
        from utils.face_detector import FaceDetector
        
        # Test Haar cascade initialization (should work with OpenCV built-in)
        detector = FaceDetector(method="haar")
        print("  ✓ Haar cascade detector initialized")
        
        # Test with dummy frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = detector.detect_faces(dummy_frame)
        print(f"  ✓ Face detection on dummy frame: {len(faces)} faces")
        
        # Test DNN (may not have models, but should initialize without crashing)
        detector_dnn = FaceDetector(method="dnn", dnn_proto="models/deploy.prototxt",
                                   dnn_model="models/res10_300x300_ssd_iter_140000.caffemodel")
        print("  ✓ DNN detector initialized (may use fallback)")
        
        return True
    except Exception as e:
        print(f"  ✗ Face detector test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """Test logging functionality"""
    print("\nTesting logger...")
    try:
        from utils.logger import TheftLogger
        import os
        
        # Create logger with test directory
        test_log_dir = "/tmp/test_logs"
        test_evidence_dir = "/tmp/test_evidence"
        logger = TheftLogger(test_log_dir, test_evidence_dir)
        print("  ✓ Logger initialized")
        
        # Test logging an event
        object_info = {
            "track_id": 42,
            "class_name": "test_object",
            "confidence": 0.95,
            "bbox": (100, 100, 200, 200),
            "displacement": 150.5
        }
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        event_id = logger.log_theft_event("test_displacement", object_info, dummy_frame, [])
        print(f"  ✓ Event logged: {event_id}")
        
        # Check if files were created
        assert os.path.exists(test_log_dir), "Log directory not created"
        print("  ✓ Log files created")
        
        return True
    except Exception as e:
        print(f"  ✗ Logger test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracked_object():
    """Test TrackedObject class"""
    print("\nTesting TrackedObject...")
    try:
        from theft_detector import TrackedObject
        import config
        
        # Create tracked object
        obj = TrackedObject(track_id=1, class_name="bottle", 
                          bbox=(100, 100, 200, 200), confidence=0.85)
        print("  ✓ TrackedObject created")
        
        # Update position
        obj.update((150, 150, 250, 250), 0.90)
        print(f"  ✓ Object updated, displacement: {obj.get_displacement():.1f}px")
        
        # Test velocity calculation
        assert obj.velocity != (0.0, 0.0), "Velocity should be non-zero after movement"
        print(f"  ✓ Velocity calculated: {obj.velocity}")
        
        # Test missing frames
        obj.mark_missing()
        assert obj.missing_frames == 1, "Missing frames counter incorrect"
        print("  ✓ Missing frames tracking works")
        
        # Test predicted bbox
        pred_bbox = obj.get_current_bbox()
        assert pred_bbox is not None, "Predicted bbox should not be None"
        print(f"  ✓ Predicted bbox: {pred_bbox}")
        
        # Test out of scope detection
        is_out = obj.check_out_of_scope(1280, 720)
        print(f"  ✓ Out of scope check: {is_out}")
        
        # Test nearby person tracking
        obj.add_nearby_person(42, 100)
        assert len(obj.nearby_persons_history) == 1, "Nearby person should be recorded"
        print("  ✓ Nearby person history tracking works")
        
        return True
    except Exception as e:
        print(f"  ✗ TrackedObject test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracked_person():
    """Test TrackedPerson class"""
    print("\nTesting TrackedPerson...")
    try:
        from theft_detector import TrackedPerson
        
        # Create tracked person
        person = TrackedPerson(track_id=1, bbox=(100, 100, 200, 300), confidence=0.90)
        print("  ✓ TrackedPerson created")
        
        # Test position update and velocity
        person.update((110, 105, 210, 305), 0.92)
        assert person.velocity != (0.0, 0.0), "Velocity should be calculated"
        print(f"  ✓ Person velocity: {person.velocity}")
        
        # Test position prediction
        person.mark_missing()
        pred_pos = person.predict_position()
        assert pred_pos is not None, "Predicted position should not be None"
        print(f"  ✓ Predicted position: {pred_pos}")
        
        # Test thief marking
        person.mark_as_thief(100)
        assert person.is_thief == True, "Person should be marked as thief"
        print("  ✓ Thief marking works")
        
        return True
    except Exception as e:
        print(f"  ✗ TrackedPerson test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_frame_buffer():
    """Test FrameBuffer class for multithreading"""
    print("\nTesting FrameBuffer...")
    try:
        from theft_detector import FrameBuffer
        import numpy as np
        
        buffer = FrameBuffer(maxsize=3)
        print("  ✓ FrameBuffer created")
        
        # Test putting frames
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 128
        
        buffer.put(frame1, 1)
        buffer.put(frame2, 2)
        print("  ✓ Frames added to buffer")
        
        # Test getting frames
        result = buffer.get(timeout=0.1)
        assert result is not None, "Should get frame from buffer"
        assert result[1] == 1, "Should get first frame"
        print("  ✓ Frame retrieval works")
        
        # Test get latest
        latest = buffer.get_latest()
        assert latest is not None, "Latest frame should exist"
        assert latest[1] == 2, "Latest should be frame 2"
        print("  ✓ Get latest frame works")
        
        # Test clear
        buffer.clear()
        empty_result = buffer.get(timeout=0.1)
        assert empty_result is None, "Buffer should be empty after clear"
        print("  ✓ Buffer clear works")
        
        return True
    except Exception as e:
        print(f"  ✗ FrameBuffer test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    try:
        import config
        
        # Check essential config values exist
        assert hasattr(config, 'DISPLACEMENT_THRESHOLD'), "Missing DISPLACEMENT_THRESHOLD"
        assert hasattr(config, 'DISAPPEARANCE_FRAMES'), "Missing DISAPPEARANCE_FRAMES"
        assert hasattr(config, 'YOLO_MODEL'), "Missing YOLO_MODEL"
        print("  ✓ Configuration loaded successfully")
        print(f"    - Displacement threshold: {config.DISPLACEMENT_THRESHOLD}px")
        print(f"    - Disappearance frames: {config.DISAPPEARANCE_FRAMES}")
        print(f"    - YOLO model: {config.YOLO_MODEL}")
        
        # Check new tracking optimization config
        assert hasattr(config, 'TRACK_BUFFER'), "Missing TRACK_BUFFER"
        assert hasattr(config, 'PREDICTION_FRAMES'), "Missing PREDICTION_FRAMES"
        assert hasattr(config, 'OUT_OF_SCOPE_FRAMES'), "Missing OUT_OF_SCOPE_FRAMES"
        print(f"    - Track buffer: {config.TRACK_BUFFER} frames")
        print(f"    - Prediction frames: {config.PREDICTION_FRAMES}")
        print(f"    - Out of scope frames: {config.OUT_OF_SCOPE_FRAMES}")
        
        # Check multithreading config
        assert hasattr(config, 'ENABLE_MULTITHREADING'), "Missing ENABLE_MULTITHREADING"
        assert hasattr(config, 'FRAME_QUEUE_SIZE'), "Missing FRAME_QUEUE_SIZE"
        print(f"    - Multithreading: {'Enabled' if config.ENABLE_MULTITHREADING else 'Disabled'}")
        print(f"    - Frame queue size: {config.FRAME_QUEUE_SIZE}")
        
        # Verify YOLOv8m model is configured
        assert 'yolov8m' in config.YOLO_MODEL.lower(), "Should use YOLOv8m model"
        print("  ✓ YOLOv8m model configured")
        
        return True
    except Exception as e:
        print(f"  ✗ Config test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("Safetronics Theft Detection System - Component Tests")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Geometry Utils", test_geometry),
        ("Face Detector", test_face_detector),
        ("Logger", test_logger),
        ("Tracked Object", test_tracked_object),
        ("Tracked Person", test_tracked_person),
        ("Frame Buffer", test_frame_buffer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
