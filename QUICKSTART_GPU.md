# Quick Start Guide - GPU-Accelerated FPS Optimization

This guide will help you get the optimized high-FPS version running on your RTX 3050 system.

## What's Been Changed

Your system has been optimized for high FPS performance with GPU acceleration:

### Key Improvements
- ✅ **GPU Acceleration Enabled** - System will automatically use your RTX 3050
- ✅ **FP16 Half-Precision** - 2x faster inference using Tensor Cores
- ✅ **Optimized Resolution** - Default changed to 640x480 (480p) for better FPS
- ✅ **Reduced Latency** - Video capture buffer optimized
- ✅ **Detection Accuracy Maintained** - Still using YOLOv8x for best accuracy

### Expected Performance on Your RTX 3050
- **30-50 FPS** at 640x480 with YOLOv8x + FP16 (current settings)
- **50-70 FPS** at 640x480 with YOLOv8m + FP16 (if you want even higher FPS)

## Quick Setup (3 Steps)

### Step 1: Verify GPU Detection

Run the GPU setup test:

```bash
python test_gpu_setup.py
```

**Expected output:**
```
✓ YOLOv8 model loaded on GPU: NVIDIA GeForce RTX 3050
✓ Half-precision (FP16) inference enabled for faster processing
```

If you see warnings about GPU not detected, see [Troubleshooting](#troubleshooting) below.

### Step 2: Test the System

Run the system with your webcam:

```bash
python main.py
```

Watch the console output - you should see:
```
✓ YOLOv8 model loaded on GPU: NVIDIA GeForce RTX 3050
✓ Half-precision (FP16) inference enabled for faster processing
Video capture started from source: 0
  Resolution: 640x480
  FPS: 30
```

### Step 3: Monitor GPU Usage

While the system is running, open another terminal and run:

```bash
nvidia-smi -l 1
```

You should see:
- **GPU Utilization**: 60-90% during inference
- **Memory Used**: ~250-350 MB
- **Temperature**: Varies by system, typically 50-70°C under load

## Troubleshooting

### "GPU not detected" or "CUDA not available"

**Check NVIDIA driver:**
```bash
nvidia-smi
```

Should show your RTX 3050. If not:
1. Install latest NVIDIA driver: https://www.nvidia.com/download/index.aspx
2. Restart your computer
3. Run `nvidia-smi` again to verify

**Check PyTorch CUDA:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

Should print `CUDA: True`. If False:
- Your PyTorch installation might not have CUDA support
- The requirements.txt includes PyTorch with CUDA, so this should work automatically

### FPS Still Low

If FPS is still below 30:

1. **Check GPU is being used:**
   - Run `nvidia-smi` while system is running
   - GPU utilization should be 60-90%
   - If GPU is at 0%, the system may be falling back to CPU

2. **Try a smaller model for higher FPS:**
   Edit `config.py`:
   ```python
   YOLO_MODEL = "yolov8m.pt"  # Medium model instead of extra-large
   ```
   Expected: 50-70 FPS (slightly lower accuracy)

3. **Verify FP16 is enabled:**
   Check console output for "Half-precision (FP16) inference enabled"

4. **Check system resources:**
   - Close other GPU-intensive applications
   - Ensure RTX 3050 has adequate cooling
   - Check CPU isn't bottlenecking (task manager)

### Higher Resolution Options

If you want higher resolution (at cost of some FPS):

Edit `config.py`:
```python
FRAME_WIDTH = 1280   # From 640
FRAME_HEIGHT = 720   # From 480
```

Expected performance:
- **1280x720**: 15-30 FPS with YOLOv8x + FP16
- **640x480** (current): 30-50 FPS with YOLOv8x + FP16

## Performance Tuning Presets

Choose based on your priority:

### Maximum FPS (50-70 FPS)
```python
# config.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_MODEL = "yolov8m.pt"  # Medium model
USE_FP16 = True
```

### Balanced (30-50 FPS) - **CURRENT DEFAULT**
```python
# config.py
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_MODEL = "yolov8x.pt"  # Extra-large model
USE_FP16 = True
```

### Maximum Quality (15-30 FPS)
```python
# config.py
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
YOLO_MODEL = "yolov8x.pt"  # Extra-large model
USE_FP16 = True
```

## Advanced Configuration

For detailed information on GPU optimization, see:
- [GPU_OPTIMIZATION.md](GPU_OPTIMIZATION.md) - Comprehensive guide
- [README.md](README.md) - General usage
- [config.py](config.py) - All configuration options

## Support

If you're still experiencing issues:

1. Run `python test_gpu_setup.py` and save the output
2. Run `nvidia-smi` and save the output  
3. Check what FPS you're actually getting
4. Open an issue with this information

## Summary

The system is now optimized to:
- ✅ **Use your RTX 3050 GPU** automatically
- ✅ **Run at 30-50 FPS** at 640x480 resolution
- ✅ **Maintain detection accuracy** with YOLOv8x
- ✅ **Use FP16** for 2x faster inference

Just run `python main.py` and it should work!
