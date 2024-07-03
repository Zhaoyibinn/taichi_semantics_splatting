import torch
import torch.nn as nn
from dataclasses import dataclass
from pytorch_msssim import ssim
from dataclass_wizard import YAMLWizard


class LossFunction(nn.Module):
    @dataclass
    class LossFunctionConfig(YAMLWizard):
        lambda_value: float = 0.2
        enable_regularization: bool = True
        regularization_weight: float = 2


    def __init__(self, config: LossFunctionConfig):
        super().__init__()
        self.iteration = 0
        self.config = config
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Sigmoid())
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=3,
                      kernel_size=5,
                      stride=1,
                      padding=2),
            nn.Sigmoid())

    def forward(self, iteration,predicted_image, ground_truth_image, point_invalid_mask=None, pointcloud_features=None):
        """
        L = (1 − 𝜆)L1 + 𝜆LD-SSIM
        predicted_image: (B, C, H, W) or (C, H, W)
        ground_truth_image: (B, C, H, W) or (C, H, W)
        """
        self.iteration = iteration
        if len(predicted_image.shape) == 3:
            predicted_image = predicted_image.unsqueeze(0)
        if len(ground_truth_image.shape) == 3:
            ground_truth_image = ground_truth_image.unsqueeze(0)

        # predicted_image = torch.tensor(predicted_image, requires_grad=True)

        if self.iteration >= 99999:

            x = self.conv5(predicted_image)
            x = self.conv5(x)
            x = self.conv5(x)
            x = self.conv5(x)
            # x = self.conv3(x)
            mask = self.conv3(x)

            # outlayer = nn.Sigmoid()
            # mask_sig =outlayer((mask * 10))
            # mask01 = mask>0.5
            # mask_sig = torch.where(mask01, 1.0, 0.0)
            mask_sig = mask
            mask1 = torch.ones_like(mask_sig)
            # if self.iteration % 1000 == 0:#可视化mask
            #     import numpy as np
            #     import cv2
            #     pic = np.asarray(mask_sig[0][0].detach().cpu())*255
            #     cv2.imwrite("mask.png",pic)

            # 为mask准备的loss
            mask_loss_fun = nn.MSELoss()
            image_mlp = mask_sig * predicted_image.detach()
            ground_truth_image_mlp = mask_sig * ground_truth_image.detach()
            mask_size_loss = 0.05 * mask_loss_fun(mask_sig.float(),mask1.float())
            after_mask_loss = torch.abs(image_mlp - ground_truth_image_mlp).mean()
            # mask_size_loss = mask_size_loss/mask_size_loss.detach()
            # after_mask_loss = after_mask_loss / after_mask_loss.detach()
            mask_loss = mask_size_loss + after_mask_loss

            # 为高斯准备的loss
            image_mlp = mask_sig.detach() * predicted_image
            ground_truth_image_mlp = mask_sig.detach() * ground_truth_image
        else:
            image_mlp = predicted_image
            ground_truth_image_mlp = ground_truth_image
            mask_loss = nn.Parameter(torch.empty(1).to("cuda"))
            # print(self.iteration)
        LD_SSIM = 1 - ssim(predicted_image, ground_truth_image,data_range=1, size_average=True)
        L1 = torch.abs(predicted_image - ground_truth_image).mean()
        # L1 = torch.abs(predicted_image - ground_truth_image).mean()
        L = (1 - self.config.lambda_value) * L1 + \
            self.config.lambda_value * LD_SSIM
        if pointcloud_features is not None and self.config.enable_regularization:
            regularization_loss = self._regularization_loss(point_invalid_mask, pointcloud_features)
            L = L + self.config.regularization_weight * regularization_loss
        return L, L1, LD_SSIM,mask_loss

    def _regularization_loss(self, point_invalid_mask, pointcloud_features):
        """ add regularization loss to pointcloud_features, especially for s.
        exp(s) is the length of three-major axis of the ellipsoid. we don't want
        it to be too large. first we try L2 regularization.

        Args:
            pointcloud_features (_type_): _description_
        """
        s = pointcloud_features[point_invalid_mask == 0, 4:7]
        exp_s = torch.exp(s)
        regularization_loss = torch.norm(exp_s, dim=1).mean()
        return regularization_loss
        
        

