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
        
        # Create tracked object
        obj = TrackedObject(track_id=1, class_name="bottle", 
                          bbox=(100, 100, 200, 200), confidence=0.85)
        print("  ✓ TrackedObject created")
        
        # Update position
        obj.update((150, 150, 250, 250), 0.90)
        print(f"  ✓ Object updated, displacement: {obj.get_displacement():.1f}px")
        
        # Test missing frames
        obj.mark_missing()
        assert obj.missing_frames == 1, "Missing frames counter incorrect"
        print("  ✓ Missing frames tracking works")
        
        return True
    except Exception as e:
        print(f"  ✗ TrackedObject test error: {e}")
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
