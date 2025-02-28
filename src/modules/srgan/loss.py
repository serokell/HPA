"""
Adaptation from https://github.com/leftthomas/SRGAN
"""

import torch
from torch import nn
from torchvision.models.vgg import vgg16

class GeneratorLoss(nn.Module):
    def __init__(self, out_channels=1, perception_enabled=True, **kwargs):
        super().__init__()
        self.perception_enabled = perception_enabled
        
        if self.perception_enabled:
            vgg = vgg16(pretrained=True)
            sd = vgg.features[0].state_dict()
            sd['weight'] = torch.cat(
                [torch.unsqueeze(sd['weight'].sum(1), 1)] * out_channels, dim=1)
            conv2d = nn.Conv2d(out_channels, 64, 3, padding=1)
            conv2d.load_state_dict(sd)
            loss_network = nn.Sequential(conv2d, *list(vgg.features)[1:31]).eval()
            for param in loss_network.parameters():
                param.requires_grad = False
            self.loss_network = loss_network
            
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        if self.perception_enabled:
            perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        
        if self.perception_enabled:
            return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss
        return image_loss + 0.001 * adversarial_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]