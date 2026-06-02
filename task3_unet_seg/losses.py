import torch
from torch import nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=255, eps=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        valid = target != self.ignore_index
        safe_target = target.clone()
        safe_target[~valid] = 0

        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dims)
        union = probs.sum(dims) + one_hot.sum(dims)
        dice = (2 * intersection + self.eps) / (union + self.eps)
        present = one_hot.sum(dims) > 0
        if present.any():
            dice = dice[present]
        return 1 - dice.mean()


def build_loss(name, num_classes, ignore_index=255):
    name = name.lower()
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    if name == "ce":
        return ce
    if name == "dice":
        return dice
    if name == "ce_dice":
        return lambda logits, target: ce(logits, target) + dice(logits, target)
    raise ValueError("loss must be one of: ce, dice, ce_dice")
