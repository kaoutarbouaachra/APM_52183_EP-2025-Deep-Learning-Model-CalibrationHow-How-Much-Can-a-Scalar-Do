import torch
import torch.nn as nn
import timm


class MLPMixer(nn.Module):
    def __init__(self, num_classes=10, model_name="mixer_b16_224", pretrained=True):
        super().__init__()

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)