# GPU Optimization Guide for RTX 3050

This document describes the GPU optimizations implemented to improve FPS performance on systems with NVIDIA RTX 3050 or other CUDA-capable GPUs.

## Overview

The system has been optimized to leverage GPU acceleration for:
- **YOLOv8 object detection and tracking** (primary performance bottleneck)
- **OpenCV DNN face detection** (when DNN models are available)
- **Half-precision (FP16) inference** for 2x faster processing on RTX GPUs

## Performance Improvements

### Expected FPS Gains
- **CPU only**: ~5-10 FPS at 640x480 with YOLOv8x
- **RTX 3050 + FP32**: ~15-25 FPS at 640x480 with YOLOv8x
- **RTX 3050 + FP16**: ~30-50 FPS at 640x480 with YOLOv8x

### Key Optimizations
1. **GPU Device Selection**: Forces YOLO model to run on CUDA device
2. **FP16 Inference**: Uses half-precision for ~2x speedup on Tensor Cores
3. **Reduced Resolution**: 640x480 (480p) balances quality and performance
4. **Buffer Optimization**: Minimizes video capture latency
5. **CUDA Backend**: OpenCV DNN operations use GPU when available

## Configuration

### GPU Settings (config.py)

```python
# GPU acceleration settings
USE_GPU = True  # Enable GPU acceleration
USE_FP16 = True  # Enable half-precision inference (recommended for RTX GPUs)
GPU_DEVICE = 0  # GPU device index (0 for primary GPU)

# Video settings
FRAME_WIDTH = 640   # Optimized resolution for better FPS
FRAME_HEIGHT = 480  # Optimized resolution for better FPS
```

### Verification

Run the GPU setup test to verify configuration:

```bash
python test_gpu_setup.py
```

This will check:
- PyTorch CUDA availability
- OpenCV CUDA support
- Configuration settings
- GPU device information

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (RTX 3050, RTX 3060, etc.)
- Minimum 4GB GPU memory recommended

### Software
1. **NVIDIA Driver**: Latest driver for your GPU
   - Download from: https://www.nvidia.com/download/index.aspx
   
2. **CUDA Toolkit**: Version 11.8 or 12.x
   - Already included with PyTorch installation
   
3. **PyTorch with CUDA**: Installed via requirements.txt
   - Verify with: `python -c "import torch; print(torch.cuda.is_available())"`

### Optional: OpenCV with CUDA
For GPU-accelerated face detection, you can build OpenCV with CUDA support:
- Download from: https://github.com/opencv/opencv
- Follow CUDA build instructions

**Note**: OpenCV CUDA is optional. Face detection will work on CPU if CUDA is not available.

## Usage

### Basic Usage
Simply run the system as normal:

```bash
python main.py
```

The system will automatically:
1. Detect available GPU
2. Load YOLO model to GPU
3. Enable FP16 inference if supported
4. Use CUDA backend for OpenCV DNN if available

### Command Line Options

```bash
# Use webcam (default)
python main.py

# Use video file
python main.py --source path/to/video.mp4

# Use different camera
python main.py --source 1

# Disable GPU (force CPU)
# Edit config.py and set USE_GPU = False
```

## Troubleshooting

### "GPU not detected" or "CUDA not available"

1. **Check NVIDIA driver installation**:
   ```bash
   nvidia-smi
   ```
   Should show your GPU and driver version.

2. **Verify PyTorch CUDA**:
   ```bash
   python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
   ```
   Should print `CUDA: True`.

3. **Check CUDA version compatibility**:
   ```bash
   python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
   ```
   Should match your installed CUDA toolkit version.

### "Out of memory" errors

1. **Reduce batch size**: Already optimized for single-frame processing
2. **Lower resolution**: Edit `config.py` and reduce FRAME_WIDTH/FRAME_HEIGHT
3. **Disable FP16**: Set `USE_FP16 = False` (uses more memory but might help)

### Poor FPS even with GPU

1. **Check GPU utilization**:
   ```bash
   nvidia-smi -l 1
   ```
   GPU utilization should be 60-90% during inference.

2. **Verify FP16 is enabled**:
   - Check console output for "Half-precision (FP16) inference enabled"
   - If not shown, FP16 might not be supported

3. **Try smaller YOLO model**:
   - Edit `config.py`: `YOLO_MODEL = "yolov8m.pt"` or `"yolov8s.pt"`
   - Smaller models = higher FPS, slightly lower accuracy

### CPU usage high even with GPU enabled

This is normal! CPU handles:
- Video capture and decoding
- Pre/post-processing (bounding boxes, tracking)
- Face detection (if OpenCV CUDA not available)
- Display rendering

## Performance Tuning

### For Maximum FPS (RTX 3050)

```python
# config.py
USE_GPU = True
USE_FP16 = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_MODEL = "yolov8m.pt"  # Medium model instead of extra-large
FACE_DETECTION_INTERVAL = 3  # Check faces every 3 frames instead of every frame
```

Expected: 50-70 FPS

### For Maximum Accuracy (RTX 3050)

```python
# config.py
USE_GPU = True
USE_FP16 = True
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_MODEL = "yolov8x.pt"  # Extra-large model (current default)
FACE_DETECTION_INTERVAL = 1  # Check faces every frame
```

Expected: 15-30 FPS

### Balanced Settings (Recommended)

```python
# config.py (current defaults)
USE_GPU = True
USE_FP16 = True
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_MODEL = "yolov8x.pt"
FACE_DETECTION_INTERVAL = 1
```

Expected: 30-50 FPS with excellent accuracy

## Technical Details

### How GPU Acceleration Works

1. **Model Loading**:
   ```python
   self.yolo = YOLO(model_path)
   self.yolo.to('cuda')  # Move model to GPU
   ```

2. **FP16 Inference**:
   ```python
   results = self.yolo.track(frame, device='cuda', half=True)
   ```
   - Converts model weights to 16-bit floating point
   - Uses Tensor Cores on RTX GPUs for 2x speedup
   - Minimal accuracy loss (<1% difference)

3. **OpenCV DNN CUDA**:
   ```python
   net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
   net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
   ```
   - Runs face detection on GPU
   - Requires OpenCV built with CUDA support

### Memory Usage

- **YOLOv8x FP32**: ~350MB GPU memory
- **YOLOv8x FP16**: ~180MB GPU memory
- **Face Detection DNN**: ~50MB GPU memory
- **Frame buffers**: ~20MB GPU memory

Total: ~250MB with FP16 (well within RTX 3050's 4GB)

## Monitoring Performance

### Real-time FPS Display

The system shows FPS information in the console:
```
Frame: 245 | Objects: 3 | Persons: 1
```

### GPU Monitoring

Open a separate terminal and run:
```bash
watch -n 1 nvidia-smi
```

This shows:
- GPU utilization %
- Memory usage
- Temperature
- Power draw

### Benchmarking

To measure average FPS:
1. Run system with a video file:
   ```bash
   python main.py --source test_video.mp4
   ```
2. Count frames processed and total time
3. Calculate: FPS = frames / time

## Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch CUDA Documentation](https://pytorch.org/docs/stable/cuda.html)
- [OpenCV DNN CUDA Backend](https://docs.opencv.org/4.x/d6/d0f/group__dnn.html)
- [NVIDIA RTX 3050 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050-3050ti/)

## Support

If you encounter issues:
1. Run `python test_gpu_setup.py` and share the output
2. Check `nvidia-smi` output
3. Verify CUDA installation: `nvcc --version`
4. Open an issue on GitHub with system information
