"""
Geometry utility functions for bounding box operations
"""
import numpy as np
from typing import Tuple, List


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes
    
    Args:
        box1: (x1, y1, x2, y2) coordinates
        box2: (x1, y1, x2, y2) coordinates
        
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area


def calculate_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Calculate center point of a bounding box
    
    Args:
        box: (x1, y1, x2, y2) coordinates
        
    Returns:
        (cx, cy) center coordinates
    """
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(point1: Tuple[float, float], 
                       point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: (x, y) coordinates
        point2: (x, y) coordinates
        
    Returns:
        Distance between points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert box format from (x, y, w, h) to (x1, y1, x2, y2)
    
    Args:
        box: (x, y, w, h) format
        
    Returns:
        (x1, y1, x2, y2) format
    """
    x, y, w, h = box
    return (x, y, x + w, y + h)


def xyxy_to_xywh(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """
    Convert box format from (x1, y1, x2, y2) to (x, y, w, h)
    
    Args:
        box: (x1, y1, x2, y2) format
        
    Returns:
        (x, y, w, h) format
    """
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)


def is_point_in_box(point: Tuple[float, float], 
                    box: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is inside a bounding box
    
    Args:
        point: (x, y) coordinates
        box: (x1, y1, x2, y2) bounding box
        
    Returns:
        True if point is inside box
    """
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2
