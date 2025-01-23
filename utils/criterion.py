import torch
import torch.nn as nn
from torchvision import transforms as T

class DepthSegLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.lambd = args.lambd
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.option = args.option

    def forward(self, preds, depth_gt,segment_gt=None):
        #gausian = T.GaussianBlur(7, sigma=0.3)
        if self.option == "depth":
            pred_d = preds['pred_d'].squeeze()
            pred_s = None

            depth_gt = depth_gt.squeeze()
            pred_d = pred_d.squeeze()
            loss_d = self.mse_loss(pred_d, depth_gt)
            loss_s = torch.tensor(0.0)
        else:
            pred_d = preds['pred_d'].squeeze()
            pred_s = preds['pred_s'].squeeze()

            depth_gt = depth_gt.squeeze()
            pred_d = pred_d.squeeze()

            loss_d = self.mse_loss(pred_d, depth_gt)
            loss_s = self.seg_loss(pred_s,segment_gt)

        loss = loss_d + (self.lambd*loss_s)

        return loss, loss_d, loss_s

