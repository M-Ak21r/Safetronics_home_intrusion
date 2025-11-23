# Logs Directory

This directory stores JSON logs of theft detection events.

## Log Format

Each log file is named with a timestamp: `theft_log_YYYYMMDD_HHMMSS.json`

### Structure
```json
{
  "events": [
    {
      "event_id": "20231123_143022_123456",
      "timestamp": "2023-11-23T14:30:22.123456",
      "event_type": "displacement|disappearance",
      "object": {
        "track_id": 42,
        "class_name": "backpack",
        "confidence": 0.89,
        "bbox": [100, 150, 300, 350],
        "displacement": 250.5
      },
      "faces_detected": 2,
      "evidence": {
        "frame": "evidence/frames/frame_20231123_143022_123456.jpg",
        "faces": [
          "evidence/faces/face_20231123_143022_123456_0.jpg",
          "evidence/faces/face_20231123_143022_123456_1.jpg"
        ]
      }
    }
  ]
}
```

## Event Types

- **displacement**: Object moved more than the threshold distance
- **disappearance**: Object was not detected for the specified number of frames

Log files are automatically created and managed by the system.
