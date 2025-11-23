#!/usr/bin/env python3
"""
Main entry point for the Safetronics Home Intrusion Detection System

This application performs real-time theft detection using:
- YOLOv8 for object detection and tracking
- Face detection (DNN or Haar Cascade) for identifying nearby persons
- Displacement and disappearance detection for theft events
- Automatic logging of theft events with evidence images
"""
import argparse
import sys
from theft_detector import TheftDetector
import config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Safetronics Home Intrusion Detection System"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Video source (camera index number or path to video file). Default: 0 (webcam)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to YOLOv8 model. Default: {config.YOLO_MODEL}"
    )
    
    parser.add_argument(
        "--displacement-threshold",
        type=float,
        default=None,
        help=f"Displacement threshold in pixels. Default: {config.DISPLACEMENT_THRESHOLD}"
    )
    
    parser.add_argument(
        "--disappearance-frames",
        type=int,
        default=None,
        help=f"Frames before object disappearance triggers alert. Default: {config.DISAPPEARANCE_FRAMES}"
    )
    
    parser.add_argument(
        "--face-method",
        type=str,
        choices=["dnn", "haar"],
        default=None,
        help=f"Face detection method. Default: {config.FACE_DETECTION_METHOD}"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable video display window"
    )
    
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help=f"YOLO detection confidence threshold. Default: {config.CONFIDENCE_THRESHOLD}"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Update config with command line arguments
    if args.displacement_threshold is not None:
        config.DISPLACEMENT_THRESHOLD = args.displacement_threshold
    
    if args.disappearance_frames is not None:
        config.DISAPPEARANCE_FRAMES = args.disappearance_frames
    
    if args.face_method is not None:
        config.FACE_DETECTION_METHOD = args.face_method
    
    if args.no_display:
        config.SHOW_DISPLAY = False
    
    if args.confidence is not None:
        config.CONFIDENCE_THRESHOLD = args.confidence
    
    # Determine video source
    video_source = config.VIDEO_SOURCE
    if args.source is not None:
        # Try to convert to int for camera index
        try:
            video_source = int(args.source)
        except ValueError:
            # It's a file path
            video_source = args.source
    
    # Create and run detector
    print("Initializing Safetronics Theft Detection System...")
    print(f"Configuration:")
    print(f"  - Video source: {video_source}")
    print(f"  - Model: {args.model or config.YOLO_MODEL}")
    print(f"  - Displacement threshold: {config.DISPLACEMENT_THRESHOLD}px")
    print(f"  - Disappearance frames: {config.DISAPPEARANCE_FRAMES}")
    print(f"  - Face detection: {config.FACE_DETECTION_METHOD}")
    print(f"  - Confidence threshold: {config.CONFIDENCE_THRESHOLD}")
    print()
    
    try:
        detector = TheftDetector(
            model_path=args.model,
            video_source=video_source
        )
        detector.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
