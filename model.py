import torch
import torch.nn as nn
import torchvision
from torchvision import ops

class TwoStageDetector(nn.Module):
    def __init__(self, img_size, out_size, out_c, n_classes, roi_size):
        super(TwoStageDetector, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(model.children())[:8])
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x, gt_bboxes=None, gt_classes=None):
        out = self.backbone(x)
        # Implement the forward pass based on your needs
        return out
