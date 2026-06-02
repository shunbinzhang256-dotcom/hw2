import torch


@torch.no_grad()
def mean_iou(logits, target, num_classes, ignore_index=255):
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    ious = []
    for cls in range(num_classes):
        pred_c = (pred == cls) & valid
        target_c = (target == cls) & valid
        union = pred_c.logical_or(target_c).sum().item()
        if union == 0:
            continue
        inter = pred_c.logical_and(target_c).sum().item()
        ious.append(inter / union)
    return sum(ious) / len(ious) if ious else 0.0


@torch.no_grad()
def pixel_accuracy(logits, target, ignore_index=255):
    pred = logits.argmax(dim=1)
    valid = target != ignore_index
    total = valid.sum().item()
    if total == 0:
        return 0.0
    return ((pred == target) & valid).sum().item() / total
