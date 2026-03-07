from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torchvision.models as tvm


@dataclass(frozen=True)
class ResNetConfig:
    arch: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"] = "resnet152"
    pretrained: bool = True


def _load_torchvision_resnet(arch: str, pretrained: bool) -> nn.Module:
    """
    Loads a torchvision ResNet with ImageNet-1K pretrained weights.
    """
    if not hasattr(tvm, arch):
        raise ValueError(f"Unknown torchvision ResNet architecture: {arch}")

    ctor = getattr(tvm, arch)

    if pretrained:
        weights_attr = f"ResNet{arch.replace('resnet', '')}_Weights"
        if not hasattr(tvm, weights_attr):
            raise RuntimeError(
                f"Cannot find weights enum {weights_attr}. "
                "Check your torchvision version."
            )
        weights_enum = getattr(tvm, weights_attr)
        weights = weights_enum.DEFAULT
        model = ctor(weights=weights)
    else:
        model = ctor(weights=None)

    return model


def _adapt_conv1_from_7x7_to_3x3(conv7: nn.Conv2d) -> nn.Conv2d:
    """
    Create a CIFAR-style conv1 (3x3, stride 1, padding 1) and initialize its weights
    from a pretrained 7x7 conv by taking the centered 3x3 crop.

    Rationale:
      - Keeps channel dimensions intact (in_channels=3, out_channels=64).
      - Avoids any expensive resizing / upsampling of input images.
    """
    if not isinstance(conv7, nn.Conv2d):
        raise TypeError("Expected nn.Conv2d for conv1.")
    if conv7.kernel_size != (7, 7):
        raise ValueError(f"Expected a 7x7 conv, got kernel_size={conv7.kernel_size}.")
    if conv7.in_channels != 3:
        raise ValueError(f"Expected in_channels=3, got {conv7.in_channels}.")

    conv3 = nn.Conv2d(
        in_channels=conv7.in_channels,
        out_channels=conv7.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=(conv7.bias is not None),
    )

    with torch.no_grad():
        # Center crop: indices 2:5 of a 7x7 kernel is the centered 3x3 region
        w7 = conv7.weight.data
        conv3.weight.data.copy_(w7[:, :, 2:5, 2:5])
        if conv7.bias is not None and conv3.bias is not None:
            conv3.bias.data.copy_(conv7.bias.data)

    return conv3


class ResNet(nn.Module):
    """
    CIFAR-friendly wrapper around an ImageNet-pretrained torchvision ResNet:

      - Input remains 32x32 (no upsample).
      - Adapt stem:
          conv1: 7x7 stride2 -> 3x3 stride1 (initialized from center-crop of pretrained conv1)
          maxpool: removed (Identity)
      - Adapt head:
          fc: -> num_classes

    Expected input: N x 3 x 32 x 32
    """
    def __init__(
        self,
        num_classes: int,
        config: ResNetConfig = ResNetConfig(),
    ):
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be >= 2 for classification.")
        self.num_classes = num_classes

        # 1) Load ImageNet-pretrained backbone
        model = _load_torchvision_resnet(config.arch, config.pretrained)

        # 2) Adapt stem for 32x32
        model.conv1 = _adapt_conv1_from_7x7_to_3x3(model.conv1)
        model.maxpool = nn.Identity()

        # 3) Adapt classifier head
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        self.model = model

        # for name, p in self.model.named_parameters():
        #     if not name.startswith("fc."):
        #         p.requires_grad = False

    # def unfreeze_all(self) -> None:
    #     """Convenience helper to switch to full fine-tuning."""
    #     for p in self.model.parameters():
    #         p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# -----------------------
# Minimal usage examples:
# -----------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    m_c10 = ResNet(num_classes=10, config=ResNetConfig("resnet152", True)).to(device)
    x_cifar = torch.randn(2, 3, 32, 32, device=device)
    y_cifar = m_c10(x_cifar)
    print("CIFAR logits:", y_cifar.shape)  # (2, 10)