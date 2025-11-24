#!/usr/bin/env python3
"""
Example usage of the Safetronics Theft Detection System

This script demonstrates how to use the theft detector with custom settings.
"""
from theft_detector import TheftDetector
import config

# Example 1: Basic usage with webcam
print("Example 1: Basic webcam detection")
print("="*60)
detector = TheftDetector(video_source=0)
# detector.run()  # Uncomment to run

# Example 2: Custom configuration
print("\nExample 2: Custom configuration")
print("="*60)
config.DISPLACEMENT_THRESHOLD = 150  # Increase threshold to reduce false positives
config.DISAPPEARANCE_FRAMES = 45     # Wait longer before disappearance alert
config.FACE_DETECTION_METHOD = "haar"  # Use Haar cascade for better performance

detector2 = TheftDetector(video_source=0)
# detector2.run()  # Uncomment to run

# Example 3: Video file analysis
print("\nExample 3: Analyzing video file")
print("="*60)
# detector3 = TheftDetector(video_source="path/to/video.mp4")
# detector3.run()  # Uncomment to run

# Example 4: Accessing logged events
print("\nExample 4: Reading logged events")
print("="*60)
from utils.logger import TheftLogger

logger = TheftLogger()
recent_events = logger.get_recent_events(count=5)

if recent_events:
    print(f"Found {len(recent_events)} recent events:")
    for i, event in enumerate(recent_events, 1):
        print(f"\n{i}. Event ID: {event['event_id']}")
        print(f"   Type: {event['event_type']}")
        print(f"   Object: {event['object']['class_name']} (#{event['object']['track_id']})")
        print(f"   Faces detected: {event['faces_detected']}")
        print(f"   Timestamp: {event['timestamp']}")
else:
    print("No theft events logged yet. Run the detector first!")

print("\n" + "="*60)
print("To run the examples, uncomment the detector.run() lines above")
print("or run the main application: python main.py")
