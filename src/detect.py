#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rugby Element Detection - Inference Script
This script runs inference using a trained YOLOv5 model on rugby images.
"""

import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random

# Define rugby classes
CLASSES = ['ball', 'line out', 'maul', 'player', 'referee', 'ruck', 'scrum']
# Define colors for visualization (one for each class)
COLORS = {
    'ball': (255, 0, 0),       # Red
    'line out': (0, 255, 0),   # Green
    'maul': (0, 0, 255),       # Blue
    'player': (255, 255, 0),   # Yellow
    'referee': (255, 0, 255),  # Magenta
    'ruck': (0, 255, 255),     # Cyan
    'scrum': (255, 255, 255)   # White
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run inference with YOLOv5 on rugby images')
    parser.add_argument('--source', type=str, required=True, help='Path to image or directory')
    parser.add_argument('--weights', type=str, default='models/best.pt', help='Path to model weights')
    parser.add_argument('--img-size', type=int, default=640, help='Inference size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()

def detect_objects(image_path, model_path, img_size, conf_thres, iou_thres):
    """
    Simulated object detection function.
    In a real implementation, this would use YOLOv5 to detect objects.
    
    Returns simulated detection results for demonstration purposes.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None, []
        
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Simulated detections 
    num_detections = random.randint(5, 15)  # Random number of detections
    detections = []
    
    for _ in range(num_detections):
        # Random class
        class_id = random.randint(0, len(CLASSES)-1)
        class_name = CLASSES[class_id]
        
        # Random confidence
        confidence = random.uniform(conf_thres, 1.0)
        
        # Random bounding box
        x1 = random.randint(0, width - 100)
        y1 = random.randint(0, height - 100)
        w = random.randint(50, min(200, width - x1))
        h = random.randint(50, min(200, height - y1))
        
        detections.append({
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, w, h]
        })
    
    return img, detections

def draw_detections(img, detections):
    """Draw bounding boxes and labels on the image."""
    result_img = img.copy()
    
    for det in detections:
        # Get detection info
        class_name = d