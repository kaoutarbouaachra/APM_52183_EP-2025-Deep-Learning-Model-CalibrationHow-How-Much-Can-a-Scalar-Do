import torch
import torch.nn as nn
import timm


class ViT(nn.Module):
    def __init__(self, num_classes=10, model_name="deit_tiny_patch16_224", pretrained=True):
        super().__init__()

        # Load ImageNet pretrained ViT without classifier
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0
        )

        # New classification head
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)

        if features.dim() == 3:
            features = features[:, 0]  

        logits = self.head(features)
        return logits