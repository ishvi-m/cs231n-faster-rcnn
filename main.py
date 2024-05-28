import numpy as np
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
from utils import *
from model import *
import os

import torch
import torchvision
from torchvision import ops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from dataset import *
from visualization import *

img_width = 640
img_height = 480
annotation_path = "data/annotations.csv"
image_dir = os.path.join("data", "images")
#name2idx = {'pad': -1, 'camel': 0, 'bird': 1}
#idx2name = {v:k for k, v in name2idx.items()}

# Create Dataset and Dataloaders
od_dataset = ObjectDetectionDataset(annotation_path, image_dir, (img_height, img_width))#, name2idx)
od_dataloader = DataLoader(od_dataset, batch_size=2)

# Grab a batch for demonstration
for img_batch, gt_bboxes_batch, gt_classes_batch in od_dataloader:
    img_data_all = img_batch
    gt_bboxes_all = gt_bboxes_batch
    gt_classes_all = gt_classes_batch
    break
    
img_data_all = img_data_all[:2]
gt_bboxes_all = gt_bboxes_all[:2]
gt_classes_all = gt_classes_all[:2]

# Display Images and Bounding Boxes
# gt_class_1 = gt_classes_all[0].long()
# gt_class_1 = [idx2name[idx.item()] for idx in gt_class_1]

# gt_class_2 = gt_classes_all[1].long()
# gt_class_2 = [idx2name[idx.item()] for idx in gt_class_2]
    
img_data_all = img_data_all[:2]
gt_bboxes_all = gt_bboxes_all[:2]
gt_classes_all = gt_classes_all[:2]

nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

fig, axes = display_img(img_data_all, fig, axes)
fig, _ = display_bbox(gt_bboxes_all[0], fig, axes[0])#, classes=gt_class_1)
fig, _ = display_bbox(gt_bboxes_all[1], fig, axes[1])#, classes=gt_class_2)

# Convolutional Backbone Network
# We will use the first 4 layers of resnet50 as our convolutional backbone
model = torchvision.models.resnet50(pretrained=True)
req_layers = list(model.children())[:8]
backbone = nn.Sequential(*req_layers)
# unfreeze all the parameters
for param in backbone.named_parameters():
    param[1].requires_grad = True
# run the image through the backbone
out = backbone(img_data_all)
out_c, out_h, out_w = out.size(dim=1), out.size(dim=2), out.size(dim=3)
print(out_c, out_h, out_w)

# Check how much the image has been down-scaled
width_scale_factor = img_width // out_w
height_scale_factor = img_height // out_h
print(height_scale_factor, width_scale_factor)

# Visualize feature maps
nrows, ncols = (1, 2)
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

filters_data =[filters[0].detach().numpy() for filters in out[:2]]

fig, axes = display_img(filters_data, fig, axes)

# # Generate and Plot Anchor Points
# anc_pts_x, anc_pts_y = gen_anc_centers(out_size=(out_h, out_w))

# # Display Grid Mapping
# # project anchor centers onto the original image
# anc_pts_x_proj = anc_pts_x.clone() * width_scale_factor 
# anc_pts_y_proj = anc_pts_y.clone() * height_scale_factor
# nrows, ncols = (1, 2)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
 
# fig, axes = display_img(img_data_all, fig, axes)
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0])
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1])

# # Create Anchor Boxes around Anchor points
# anc_scales = [2, 4, 6]
# anc_ratios = [0.5, 1, 1.5]
# n_anc_boxes = len(anc_scales) * len(anc_ratios) # number of anchor boxes for each anchor point

# anc_base = gen_anc_base(anc_pts_x, anc_pts_y, anc_scales, anc_ratios, (out_h, out_w))
# # since all the images are scaled to the same size
# # we can repeat the anchor base for all the images
# anc_boxes_all = anc_base.repeat(img_data_all.size(dim=0), 1, 1, 1, 1)
# # plot anchor boxes on a single anchor point
# nrows, ncols = (1, 2)
# fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))

# fig, axes = display_img(img_data_all, fig, axes)

# # project anchor boxes to the image
# anc_boxes_proj = project_bboxes(anc_boxes_all, width_scale_factor, height_scale_factor, mode='a2p')

# # plot anchor boxes around selected anchor points
# sp_1 = [5, 8]
# sp_2 = [12, 9]
# bboxes_1 = anc_boxes_proj[0][sp_1[0], sp_1[1]]
# bboxes_2 = anc_boxes_proj[1][sp_2[0], sp_2[1]]

# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[0], (anc_pts_x_proj[sp_1[0]], anc_pts_y_proj[sp_1[1]]))
# fig, _ = display_grid(anc_pts_x_proj, anc_pts_y_proj, fig, axes[1], (anc_pts_x_proj[sp_2[0]], anc_pts_y_proj[sp_2[1]]))
# fig, _ = display_bbox(bboxes_1, fig, axes[0])
# fig, _ = display_bbox(bboxes_2, fig, axes[1])
