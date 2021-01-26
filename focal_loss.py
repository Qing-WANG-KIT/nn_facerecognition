import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        self.gamma = gamma
        self.ce = torch.nn.NLLLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss