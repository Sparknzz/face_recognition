import torch
import torch.nn as nn

class FocalLoss():
    def __init__(self, gamma=2):
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, output, gt):
        logits = self.ce(output, gt)
        p = torch.exp(logits)
        focal = (1-p) ** self.gamma * logits

        return focal.mean()


