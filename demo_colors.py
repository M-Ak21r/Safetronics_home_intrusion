#!/usr/bin/env python3
"""
Demo script to visualize the color-coded bounding box system
This creates a sample frame showing the different box colors
"""
import cv2
import numpy as np
import config

# Create a blank image
width, height = 1280, 720
demo_frame = np.ones((height, width, 3), dtype=np.uint8) * 255

# Add title
cv2.putText(demo_frame, "Safetronics Theft Detection - Color Coding Demo", 
           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

# Demo 1: Normal Person (Green)
x1, y1, x2, y2 = 100, 150, 300, 500
cv2.rectangle(demo_frame, (x1, y1), (x2, y2), config.COLOR_PERSON, 2)
cv2.putText(demo_frame, "Person #1", (x1, y1 - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_PERSON, 2)
cv2.putText(demo_frame, "Green = Normal Person", (x1, y2 + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Demo 2: Object (Blue)
x1, y1, x2, y2 = 400, 200, 550, 350
cv2.rectangle(demo_frame, (x1, y1), (x2, y2), config.COLOR_OBJECT, 2)
cv2.putText(demo_frame, "backpack #5", (x1, y1 - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_OBJECT, 2)
cv2.putText(demo_frame, "Blue = Object", (x1, y2 + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Demo 3: Thief (Red - thick border)
x1, y1, x2, y2 = 650, 150, 850, 500
cv2.rectangle(demo_frame, (x1, y1), (x2, y2), config.COLOR_THIEF, 3)
cv2.putText(demo_frame, "Person #3 [THIEF]", (x1, y1 - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_THIEF, 2)
cv2.putText(demo_frame, "Red = Thief (thicker border)", (x1, y2 + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Demo 4: Held Object (Gray)
x1, y1, x2, y2 = 950, 250, 1100, 400
cv2.rectangle(demo_frame, (x1, y1), (x2, y2), config.COLOR_FILTERED, 2)
cv2.putText(demo_frame, "bottle #8 [Held]", (x1, y1 - 10),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_FILTERED, 2)
cv2.putText(demo_frame, "Gray = Held by Person", (x1, y2 + 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Add legend at bottom
legend_y = 600
cv2.putText(demo_frame, "How it works:", (50, legend_y),
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
cv2.putText(demo_frame, "1. Green boxes track normal persons", (70, legend_y + 35),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(demo_frame, "2. Blue boxes track objects (bags, laptops, etc.)", (70, legend_y + 65),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
cv2.putText(demo_frame, "3. When object disappears, nearby person turns RED (marked as thief)", (70, legend_y + 95),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

# Save the demo image
output_path = "color_coding_demo.jpg"
cv2.imwrite(output_path, demo_frame)
print(f"✓ Demo image saved to {output_path}")
print("\nColor Coding Summary:")
print(f"  🟢 Green (BGR: {config.COLOR_PERSON}) = Normal Person")
print(f"  🔵 Blue (BGR: {config.COLOR_OBJECT}) = Object")
print(f"  🔴 Red (BGR: {config.COLOR_THIEF}) = Thief (thick border)")
print(f"  ⚪ Gray (BGR: {config.COLOR_FILTERED}) = Held Object")

# Try to display the image (skip if no display available)
try:
    cv2.imshow("Color Coding Demo - Press any key to close", demo_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print("\nNote: Display not available in headless environment")
    print("View the saved image: color_coding_demo.jpg")
