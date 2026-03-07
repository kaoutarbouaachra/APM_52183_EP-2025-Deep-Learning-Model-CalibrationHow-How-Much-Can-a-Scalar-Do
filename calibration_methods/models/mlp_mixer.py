# import torch
# import torch.nn as nn


# class MLPMixer(nn.Module):
#     """
#     CIFAR-native MLP-Mixer (32x32), trained from scratch.

#     - Input: (N, 3, 32, 32)
#     - Patch size: 4 (=> 8x8 = 64 tokens) or 8 (=> 4x4 = 16 tokens)
#     - Output: logits (N, num_classes)
#     """
#     def __init__(
#         self,
#         num_classes=10,
#         img_size=32,
#         patch_size=4,
#         dim=256,
#         depth=8,
#         token_mlp_dim=512,
#         channel_mlp_dim=1024,
#         dropout=0.1,
#     ):
#         super().__init__()
#         assert img_size % patch_size == 0, "img_size must be divisible by patch_size"

#         self.num_patches = (img_size // patch_size) ** 2
#         patch_dim = 3 * patch_size * patch_size

#         # Patch embedding (linear projection of flattened patches)
#         self.patch_embed = nn.Linear(patch_dim, dim)

#         # Mixer blocks
#         self.blocks = nn.ModuleList([
#             _MixerBlock(
#                 num_patches=self.num_patches,
#                 dim=dim,
#                 token_mlp_dim=token_mlp_dim,
#                 channel_mlp_dim=channel_mlp_dim,
#                 dropout=dropout,
#             )
#             for _ in range(depth)
#         ])

#         self.norm = nn.LayerNorm(dim)
#         self.head = nn.Linear(dim, num_classes)

#         self.patch_size = patch_size

#     def forward(self, x):
#         # x: (N, 3, 32, 32)
#         n, c, h, w = x.shape
#         p = self.patch_size

#         # Extract non-overlapping patches: (N, C, H/p, W/p, p, p)
#         x = x.reshape(n, c, h // p, p, w // p, p)
#         # -> (N, H/p, W/p, C, p, p)
#         x = x.permute(0, 2, 4, 1, 3, 5)
#         # -> (N, num_patches, patch_dim)
#         x = x.reshape(n, self.num_patches, c * p * p)

#         # Linear embed: (N, num_patches, dim)
#         x = self.patch_embed(x)

#         for blk in self.blocks:
#             x = blk(x)

#         # Global average over tokens
#         x = self.norm(x).mean(dim=1)
#         return self.head(x)


# class _MixerBlock(nn.Module):
#     def __init__(self, num_patches, dim, token_mlp_dim, channel_mlp_dim, dropout):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.token_mlp = _TokenMLP(num_patches, token_mlp_dim, dropout)

#         self.norm2 = nn.LayerNorm(dim)
#         self.channel_mlp = _MLP(dim, channel_mlp_dim, dropout)

#     def forward(self, x):
#         # Token mixing: operates across tokens
#         y = self.norm1(x)
#         y = self.token_mlp(y)
#         x = x + y

#         # Channel mixing: operates across channels
#         y = self.norm2(x)
#         y = self.channel_mlp(y)
#         x = x + y
#         return x


# class _TokenMLP(nn.Module):
#     """
#     Token-mixing MLP: applies an MLP over the token dimension for each channel.
#     Input/Output: (N, num_patches, dim)
#     """
#     def __init__(self, num_patches, token_mlp_dim, dropout):
#         super().__init__()
#         self.mlp = _MLP(num_patches, token_mlp_dim, dropout)

#     def forward(self, x):
#         # transpose so token dimension becomes the last dim for a per-channel MLP:
#         # (N, num_patches, dim) -> (N, dim, num_patches)
#         x = x.transpose(1, 2)
#         x = self.mlp(x)
#         # back: (N, dim, num_patches) -> (N, num_patches, dim)
#         return x.transpose(1, 2)


# class _MLP(nn.Module):
#     def __init__(self, in_dim, hidden_dim, dropout):
#         super().__init__()
#         self.fc1 = nn.Linear(in_dim, hidden_dim)
#         self.act = nn.GELU()
#         self.drop1 = nn.Dropout(dropout)
#         self.fc2 = nn.Linear(hidden_dim, in_dim)
#         self.drop2 = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


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