# Models Directory

This directory is for storing model files used by the theft detection system.

## Required Models

### YOLOv8 Model
The YOLOv8 model will be automatically downloaded by ultralytics when you first run the application.
- Default: `yolov8n.pt` (nano - lightweight for Raspberry Pi)
- Alternative: `yolov8s.pt`, `yolov8m.pt`, etc. for better accuracy on more powerful hardware

### DNN Face Detection Model (Optional)
For DNN-based face detection, download these files:

1. **deploy.prototxt**
   - URL: https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt

2. **res10_300x300_ssd_iter_140000.caffemodel**
   - URL: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

Download and place these files in this `models/` directory.

### Haar Cascade (Alternative - Built-in)
The Haar Cascade face detector is built into OpenCV and doesn't require separate downloads.
It will be used automatically if DNN models are not found.

## Download Script

You can download the face detection models using these commands:

```bash
cd models

# Download DNN prototxt
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt

# Download DNN model
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

## Model Files
- `yolov8n.pt` - YOLOv8 nano model (auto-downloaded)
- `deploy.prototxt` - DNN face detector architecture
- `res10_300x300_ssd_iter_140000.caffemodel` - DNN face detector weights
- `haarcascade_frontalface_default.xml` - Haar cascade (optional, OpenCV built-in)
