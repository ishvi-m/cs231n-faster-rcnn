import os
import xml.etree.ElementTree as ET

# def parse_annotation(annotation_path, img_dir, img_size):
#     # Your implementation of parse_annotation
#     return gt_boxes_all, gt_classes_all, img_paths

# # Add other utility functions like project_bboxes, get_req_anchors, etc.

import os
import pandas as pd
import torch

def parse_annotation(csv_path, img_dir, img_size):
    """
    Parse the annotation CSV file to get ground truth bounding boxes and image paths.

    Parameters:
    - csv_path: str, path to the annotation CSV file
    - img_dir: str, directory where images are stored
    - img_size: tuple, target size for resizing images (height, width)

    Returns:
    - gt_boxes_all: list of torch.Tensor, each tensor containing bounding boxes for one image
    - img_paths: list of str, file paths to the images
    """
    # Read the CSV file
    annotations = pd.read_csv(csv_path)
    
    # Initialize lists to hold the bounding boxes and image paths
    gt_boxes_all = []
    img_paths = []

    # Group the annotations by filename
    grouped = annotations.groupby('filename')

    for filename, group in grouped:
        img_path = os.path.join(img_dir, filename)
        img_paths.append(img_path)
        
        # Extract bounding boxes
        bboxes = group[['xmin', 'ymin', 'xmax', 'ymax']].values
        gt_boxes = [torch.tensor([xmin, ymin, xmax, ymax]) for xmin, ymin, xmax, ymax in bboxes]
        
        gt_boxes_all.append(torch.stack(gt_boxes))

    return gt_boxes_all, img_paths

# Example usage
csv_path = 'path/to/annotations.csv'
img_dir = 'path/to/images'
img_size = (640, 640)  # Adjust if necessary

gt_boxes_all, img_paths = parse_annotation(csv_path, img_dir, img_size)
print(gt_boxes_all)
print(img_paths)
