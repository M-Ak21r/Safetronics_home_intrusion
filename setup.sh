#!/bin/bash
# Setup script for Safetronics Home Intrusion Detection System

echo "==========================================="
echo "Safetronics Setup Script"
echo "==========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment (optional but recommended)
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " create_venv
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated."
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies."
    exit 1
fi

echo ""
echo "Dependencies installed successfully!"

# Download face detection models
echo ""
read -p "Do you want to download DNN face detection models? (recommended for better accuracy) [y/N]: " download_models
if [[ $download_models =~ ^[Yy]$ ]]; then
    echo "Downloading face detection models..."
    cd models
    
    echo "Downloading deploy.prototxt..."
    wget -q --show-progress https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/deploy.prototxt
    
    echo "Downloading caffemodel (this may take a moment)..."
    wget -q --show-progress https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
    
    cd ..
    echo "Face detection models downloaded successfully!"
else
    echo "Skipping model download. The system will use Haar Cascade (built-in) for face detection."
fi

# Create necessary directories
echo ""
echo "Creating necessary directories..."
mkdir -p logs evidence/faces evidence/frames models

echo ""
echo "==========================================="
echo "Setup completed successfully!"
echo "==========================================="
echo ""
echo "To run the theft detection system:"
echo "  python main.py"
echo ""
echo "For help and options:"
echo "  python main.py --help"
echo ""
echo "To test the installation:"
echo "  python test_system.py"
echo ""

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Note: Remember to activate the virtual environment before running:"
    echo "  source venv/bin/activate"
fi

echo ""
echo "Happy monitoring!"
