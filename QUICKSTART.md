# Quick Start Guide

Get started with Safetronics Home Intrusion Detection in 5 minutes!

## 🚀 Fast Installation

### Option 1: Automated Setup (Recommended)
```bash
./setup.sh
```

### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p logs evidence/faces evidence/frames models

# (Optional) Download face detection models for better accuracy
cd models
wget https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
cd ..
```

## 🎥 Running the System

### Basic Usage (Webcam)
```bash
python main.py
```
Press `q` to quit.

### Using a Video File
```bash
python main.py --source path/to/video.mp4
```

### Custom Settings
```bash
python main.py \
  --displacement-threshold 150 \
  --disappearance-frames 45 \
  --face-method haar \
  --confidence 0.6
```

## 📊 Viewing Results

After running the detection:

### Check Logs
```bash
cat logs/theft_log_*.jsonl
```

### View Evidence
- **Annotated frames**: `evidence/frames/frame_*.jpg`
- **Face crops**: `evidence/faces/face_*.jpg`

### Example: Reading Logs in Python
```python
from utils.logger import TheftLogger

logger = TheftLogger()
events = logger.get_recent_events(count=10)

for event in events:
    print(f"Event: {event['event_type']}")
    print(f"Object: {event['object']['class_name']}")
    print(f"Faces: {event['faces_detected']}")
```

## ⚙️ Common Configuration

Edit `config.py` to customize:

```python
# How far an object must move (in pixels) to trigger alert
DISPLACEMENT_THRESHOLD = 100

# How many frames an object must be missing before alert
DISAPPEARANCE_FRAMES = 30

# Face detection method: "dnn" (accurate) or "haar" (fast)
FACE_DETECTION_METHOD = "dnn"

# Detection confidence (lower = more sensitive)
CONFIDENCE_THRESHOLD = 0.5
```

## 🐛 Troubleshooting

### Camera not working?
```bash
# Try different camera index
python main.py --source 1

# On Linux, check permissions
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

### Poor detection?
- Ensure good lighting
- Lower confidence: `--confidence 0.4`
- Try different YOLO model: `--model yolov8s.pt`

### Too many false positives?
- Increase displacement: `--displacement-threshold 200`
- Increase disappearance: `--disappearance-frames 60`

## 🎯 What Gets Detected?

The system tracks all objects EXCEPT persons:
- **Bags, backpacks, handbags**
- **Electronics (laptops, phones, tablets)**
- **Personal items (bottles, books, umbrellas)**
- **Furniture and valuables**

Objects held by people are automatically filtered out!

## 📱 Use Cases

- **Home Security**: Monitor entryways and living spaces
- **Office**: Protect valuable equipment
- **Retail**: Detect shoplifting
- **Warehouse**: Track inventory movement

## 🎓 Next Steps

1. **Test the system**: `python test_system.py`
2. **Try examples**: `python example_usage.py`
3. **Read full docs**: See `README.md`
4. **Customize**: Edit `config.py` for your needs

## 💡 Tips

- **Raspberry Pi**: Use `yolov8n.pt` (nano) for better performance
- **Desktop**: Use `yolov8m.pt` (medium) for better accuracy
- **Night Vision**: Ensure IR illumination or use low-light camera
- **Multiple Cameras**: Run multiple instances with different `--source`

---

**Need Help?** Check the full [README.md](README.md) or open an issue on GitHub!
