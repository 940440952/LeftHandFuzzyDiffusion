from torch import nn
import torch.nn.functional as F


class MultiScalePerceptualLoss(nn.Module):
    def __init__(self, vgg, scales=None):
        super(MultiScalePerceptualLoss, self).__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.vgg = vgg
        self.scales = scales

    def forward(self, input, target):
        loss = 0
        for scale in self.scales:
            input_scaled = F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=False)
            target_scaled = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += F.mse_loss(self.vgg(input_scaled), self.vgg(target_scaled), reduction="mean")
        return loss
