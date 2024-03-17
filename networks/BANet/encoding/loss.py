import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
import cv2
import numpy as np
class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.loss_weights = [1, 1, 1, 1, 1, 1, 1, 1]
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        self.seg_edge_loss = False
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        #score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss
        
    # def __call__(self, y_true, y_pred):
    #     a =  self.bce_loss(y_pred, y_true)
    #     b =  self.soft_dice_loss(y_true, y_pred)
    #     return a + b
    def SegmentationLoss(self,input, target):
        input, seg1, edge1, seg2, edge2, seg3, edge3, seg4, edge4 = input
        boundary = torch.abs(target.float() - F.avg_pool2d(target.float(), 3, 1, 1))
        boundary[boundary > 0] = 1
        boundary[boundary != 1] = 0
        seg1_loss = self.bce_loss(seg1, target)
        seg2_loss = self.bce_loss(seg2, target)
        seg3_loss = self.bce_loss(seg3, target)
        seg4_loss = self.bce_loss(seg4, target)
        if self.seg_edge_loss:
            seg1_loss += self.bce_loss(torch.abs(seg1 - F.avg_pool2d(seg1, 3, 1, 1)), boundary)
            seg2_loss += self.bce_loss(torch.abs(seg2 - F.avg_pool2d(seg2, 3, 1, 1)), boundary)
            seg3_loss += self.bce_loss(torch.abs(seg3 - F.avg_pool2d(seg3, 3, 1, 1)), boundary)
            seg4_loss += self.bce_loss(torch.abs(seg4 - F.avg_pool2d(seg4, 3, 1, 1)), boundary)
        edge1_loss = self.bce_loss(edge1, boundary)
        edge2_loss = self.bce_loss(edge2, boundary)
        edge3_loss = self.bce_loss(edge3, boundary)
        edge4_loss = self.bce_loss(edge4, boundary)
        ce_loss = self.bce_loss(input, target)
        dice_loss = self.soft_dice_loss(input, target)
        loss = 4 * ce_loss + 4 * dice_loss + self.loss_weights[0] * seg1_loss + \
               self.loss_weights[1] * edge1_loss + self.loss_weights[2] * seg2_loss + self.loss_weights[
                   3] * edge2_loss + \
               self.loss_weights[4] * seg3_loss + self.loss_weights[5] * edge3_loss + self.loss_weights[6] * seg4_loss + \
               self.loss_weights[7] * edge4_loss
        return loss
    def __call__(self, y_true, y_pred):
        a = self.SegmentationLoss(y_pred, y_true)
        return a
