import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, gt):

        # gt_one_hot = torch.zeros(output.size(), device='cuda')

        # gt_one_hot = gt_one_hot.scatter_(1, gt.view(-1,1).long(), 1)

        logp = self.ce(output, gt)
        p = torch.exp(-logp)
        focal = (1-p) ** self.gamma * logp

        return focal.mean()