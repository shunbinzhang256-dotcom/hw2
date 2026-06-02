import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34
from torchvision.models.resnet import BasicBlock, ResNet


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * weights


class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SEBlock(self.bn2.num_features, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def _replace_classifier(model, num_classes):
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _load_resnet18_backbone(model):
    state = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
    model.load_state_dict(state, strict=False)
    return model


def get_model(name, num_classes=102, pretrained=False):
    name = name.lower()

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_classifier(resnet18(weights=weights), num_classes)

    if name == "resnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return _replace_classifier(resnet34(weights=weights), num_classes)

    if name == "resnet18_se":
        model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=1000)
        if pretrained:
            model = _load_resnet18_backbone(model)
        return _replace_classifier(model, num_classes)

    try:
        import timm
    except ImportError as exc:
        raise ImportError("Install timm to use ViT/Swin models.") from exc

    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    return model
