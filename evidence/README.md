# Evidence Directory

This directory stores visual evidence of theft detection events.

## Structure

```
evidence/
├── faces/       # Cropped face images from theft events
└── frames/      # Annotated full frames showing the theft event
```

## File Naming

- **Frames**: `frame_YYYYMMDD_HHMMSS_FFFFFF.jpg`
- **Faces**: `face_YYYYMMDD_HHMMSS_FFFFFF_N.jpg` (where N is the face index)

All evidence files are automatically saved when a theft event is detected with nearby faces.

## Storage Management

Evidence files can accumulate quickly. Consider implementing a cleanup policy:
- Keep evidence for a specific time period (e.g., 30 days)
- Archive to external storage
- Delete after reviewing
