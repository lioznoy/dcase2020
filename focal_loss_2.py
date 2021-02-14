import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, alpha=0.4, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        eps = 1e-10
        input_soft = F.softmax(input, dim=1) + eps
        weight = torch.pow(-input_soft + 1., self.gamma)
        focal = -self.alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target * focal, dim=1)
        loss = torch.mean(loss_tmp)
        return loss
