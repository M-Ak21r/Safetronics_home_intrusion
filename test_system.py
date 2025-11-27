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
        
        from theft_detector import TheftDetector, StationaryObject, TrackedPerson
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
        from utils.geometry import xywh_to_xyxy, xyxy_to_xywh, is_point_in_box
        
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
        
        # Test point in box
        assert is_point_in_box((50, 50), box1) == True, "Point should be in box"
        assert is_point_in_box((150, 150), box1) == False, "Point should not be in box"
        print("  ✓ Point in box check")
        
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
        import config
        margin = config.HITBOX_MARGIN
        bbox = (100, 100, 200, 200)
        hitbox = (bbox[0] - margin, bbox[1] - margin, bbox[2] + margin, bbox[3] + margin)
        
        object_info = {
            "track_id": 42,
            "class_name": "test_object",
            "confidence": 0.95,
            "bbox": bbox,
            "hitbox": hitbox
        }
        
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        event_id = logger.log_theft_event("hand_in_hitbox", object_info, dummy_frame, [])
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


def test_stationary_object():
    """Test StationaryObject class"""
    print("\nTesting StationaryObject...")
    try:
        from theft_detector import StationaryObject
        import config
        
        # Create stationary object
        obj = StationaryObject(track_id=1, class_name="bottle", 
                              bbox=(100, 100, 200, 200), confidence=0.85)
        print("  ✓ StationaryObject created")
        
        # Check hitbox was created with margin
        margin = config.HITBOX_MARGIN
        expected_hitbox = (100 - margin, 100 - margin, 200 + margin, 200 + margin)
        assert obj.hitbox == expected_hitbox, f"Hitbox incorrect: {obj.hitbox}"
        print(f"  ✓ Hitbox created: {obj.hitbox}")
        
        # Update with small movement (should become stationary)
        for _ in range(config.STATIONARY_FRAMES + 5):
            obj.update((100, 100, 200, 200), 0.90)
        
        assert obj.is_stationary == True, "Object should be stationary"
        print("  ✓ Object marked as stationary after stable frames")
        
        # Test theft attempt recording
        obj.record_theft_attempt(42, 100)
        assert len(obj.theft_attempts) == 1, "Theft attempt should be recorded"
        print("  ✓ Theft attempt recording works")
        
        # Test missing frames
        obj.mark_missing()
        assert obj.missing_frames == 1, "Missing frames counter incorrect"
        print("  ✓ Missing frames tracking works")
        
        return True
    except Exception as e:
        print(f"  ✗ StationaryObject test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracked_person():
    """Test TrackedPerson class"""
    print("\nTesting TrackedPerson...")
    try:
        from theft_detector import TrackedPerson
        
        # Create tracked person
        person = TrackedPerson(track_id=1, bbox=(100, 100, 200, 400), confidence=0.90)
        print("  ✓ TrackedPerson created")
        
        # Test hand regions estimation
        assert len(person.hand_regions) == 2, "Should have 2 hand regions"
        print(f"  ✓ Hand regions estimated: {len(person.hand_regions)} regions")
        
        # Test hand in hitbox check
        test_hitbox = (50, 200, 300, 350)  # Hitbox that overlaps with hand region
        hand_in_box = person.check_hand_in_hitbox(test_hitbox)
        print(f"  ✓ Hand in hitbox check: {hand_in_box}")
        
        # Test position update
        person.update((110, 105, 210, 405), 0.92)
        assert person.last_bbox == (110, 105, 210, 405), "Position should be updated"
        print("  ✓ Position update works")
        
        # Test thief marking
        person.mark_as_thief(100)
        assert person.is_thief == True, "Person should be marked as thief"
        assert person.theft_attempts == 1, "Theft attempts should increment"
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
        
        # Test is_empty
        assert not buffer.is_empty(), "Buffer should not be empty"
        print("  ✓ is_empty check works")
        
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
        assert buffer.is_empty(), "Buffer should report empty after clear"
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
        assert hasattr(config, 'CONFIDENCE_THRESHOLD'), "Missing CONFIDENCE_THRESHOLD"
        assert hasattr(config, 'YOLO_MODEL'), "Missing YOLO_MODEL"
        print("  ✓ Configuration loaded successfully")
        print(f"    - Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
        print(f"    - YOLO model: {config.YOLO_MODEL}")
        
        # Check new stationary object config
        assert hasattr(config, 'STATIONARY_THRESHOLD'), "Missing STATIONARY_THRESHOLD"
        assert hasattr(config, 'STATIONARY_FRAMES'), "Missing STATIONARY_FRAMES"
        assert hasattr(config, 'HITBOX_MARGIN'), "Missing HITBOX_MARGIN"
        print(f"    - Stationary threshold: {config.STATIONARY_THRESHOLD}px")
        print(f"    - Stationary frames: {config.STATIONARY_FRAMES}")
        print(f"    - Hitbox margin: {config.HITBOX_MARGIN}px")
        
        # Check display config
        assert hasattr(config, 'DRAW_HITBOXES'), "Missing DRAW_HITBOXES"
        assert hasattr(config, 'DRAW_HAND_REGIONS'), "Missing DRAW_HAND_REGIONS"
        print(f"    - Draw hitboxes: {config.DRAW_HITBOXES}")
        print(f"    - Draw hand regions: {config.DRAW_HAND_REGIONS}")
        
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
    print("(Hand-in-Hitbox Detection Mode)")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Geometry Utils", test_geometry),
        ("Face Detector", test_face_detector),
        ("Logger", test_logger),
        ("Stationary Object", test_stationary_object),
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
