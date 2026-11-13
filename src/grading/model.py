import timm
import torch
import torch.nn as nn

LEVELS = 9


class TwoViewGrader(nn.Module):
    def __init__(self, backbone='resnet34', pretrained=True, drop=0.2):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, 256), nn.ReLU(), nn.Dropout(drop), nn.Linear(256, 1))
        self.bias = nn.Parameter(torch.zeros(LEVELS))

    def forward(self, front, back):
        b = front.shape[0]
        f = self.backbone(torch.cat([front, back], dim=0))
        z = torch.cat([f[:b], f[b:]], dim=1)
        score = self.head(z)
        return score + self.bias.unsqueeze(0)


def logits_to_grade(logits):
    return 1 + (torch.sigmoid(logits) > 0.5).sum(dim=1)


def grade_targets(grades):
    levels = torch.arange(1, LEVELS + 1, device=grades.device).unsqueeze(0)
    return (grades.unsqueeze(1) > levels).float()
