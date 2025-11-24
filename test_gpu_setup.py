#!/usr/bin/env python3
"""
Test script to verify GPU setup and optimizations
"""
import sys

def test_gpu_availability():
    """Test if GPU is available for PyTorch/YOLO"""
    print("="*60)
    print("GPU Setup Test")
    print("="*60)
    
    # Test PyTorch CUDA
    print("\n1. Testing PyTorch CUDA availability...")
    try:
        import torch
        print(f"   PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU device count: {torch.cuda.device_count()}")
            print(f"   GPU device name: {torch.cuda.get_device_name(0)}")
            print(f"   Current device: cuda:{torch.cuda.current_device()}")
            
            # Test memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   GPU memory: {total_mem:.2f} GB")
        else:
            print("   ⚠️  No CUDA GPU detected")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test OpenCV CUDA
    print("\n2. Testing OpenCV CUDA availability...")
    try:
        import cv2
        print(f"   OpenCV version: {cv2.__version__}")
        
        # Check if OpenCV was built with CUDA
        build_info = cv2.getBuildInformation()
        has_cuda = "CUDA" in build_info and "YES" in build_info.split("CUDA:")[1].split("\n")[0]
        
        if has_cuda:
            print(f"   OpenCV built with CUDA: Yes")
            try:
                cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
                print(f"   CUDA-enabled devices: {cuda_devices}")
                if cuda_devices > 0:
                    print(f"   ✓ OpenCV can use GPU for DNN operations")
            except:
                print(f"   OpenCV CUDA runtime not available")
        else:
            print(f"   OpenCV built with CUDA: No")
            print(f"   ⚠️  OpenCV DNN will run on CPU")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Test configuration
    print("\n3. Testing configuration settings...")
    try:
        import config
        print(f"   USE_GPU: {config.USE_GPU}")
        print(f"   USE_FP16: {config.USE_FP16}")
        print(f"   GPU_DEVICE: {config.GPU_DEVICE}")
        print(f"   Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(f"   YOLO Model: {config.YOLO_MODEL}")
        
        if config.USE_GPU:
            print(f"   ✓ GPU acceleration is ENABLED")
            if config.USE_FP16:
                print(f"   ✓ FP16 half-precision is ENABLED (2x faster on RTX GPUs)")
        else:
            print(f"   ⚠️  GPU acceleration is DISABLED in config")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    try:
        import torch
        import config
        if torch.cuda.is_available() and config.USE_GPU:
            print("✓ System is configured to use GPU acceleration")
            print("✓ Expected performance: Significantly improved FPS on RTX 3050")
            if config.USE_FP16:
                print("✓ FP16 mode enabled: ~2x faster inference")
            print("\nRecommendations for RTX 3050:")
            print("  - Resolution 640x480 should give smooth FPS (30+ FPS)")
            print("  - YOLOv8x with FP16 will run efficiently on GPU")
            print("  - Face detection DNN will use GPU if OpenCV has CUDA")
        else:
            print("⚠️  GPU not available or not configured")
            print("   The system will run on CPU with reduced FPS")
            print("\nTo enable GPU on your RTX 3050 system:")
            print("  1. Install CUDA toolkit (11.8 or 12.x)")
            print("  2. Ensure PyTorch with CUDA is installed")
            print("  3. Set USE_GPU=True in config.py (already set)")
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Could not complete summary: {e}")
        pass
    
    print("="*60)


if __name__ == "__main__":
    test_gpu_availability()
