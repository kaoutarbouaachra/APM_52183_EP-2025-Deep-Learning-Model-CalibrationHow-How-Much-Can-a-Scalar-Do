import random
import numpy as np
import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from models.resnet import *
from models.mlp_mixer import *
from models.vit import *
from train_model import *

import os

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_cifar_loaders(
    dataset_name,
    train_transform,
    val_transform,
    root="datasets",
    batch_size=128,
    num_workers=4,
    seed=42
):

    seed_everything(seed)

    g = torch.Generator()
    g.manual_seed(seed)

    if dataset_name == "cifar10":
        Dataset = datasets.CIFAR10

    elif dataset_name == "cifar100":
        Dataset = datasets.CIFAR100

    train_dataset = Dataset(root=root, train=True, download=False, transform=train_transform)
    val_dataset_full = Dataset(root=root, train=True, download=False, transform=val_transform)
    test_dataset = Dataset(root=root, train=False, download=False, transform=val_transform)

    train_set, val_set = random_split(train_dataset, [45000, 5000], generator=g)

    val_set.dataset = val_dataset_full

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    return train_loader, val_loader, test_loader


# Finetuning models 
device = "cuda" if torch.cuda.is_available() else "cpu"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# # Finetuning ResNet 152
# train_transform_resnet_cifar10 = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
# ])

# val_transform_resnet_cifar10 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
# ])

# train_transform_resnet_cifar100 = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
# ])

# val_transform_resnet_cifar100 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
# ])


# train_loader_resnet_cifar10, val_loader_resnet_cifar10, _ = get_cifar_loaders(
#     dataset_name="cifar10",
#     train_transform=train_transform_resnet_cifar10,
#     val_transform=val_transform_resnet_cifar10,
#     batch_size=256
# )

# train_loader_resnet_cifar100, val_loader_resnet_cifar100, _ = get_cifar_loaders(
#     dataset_name="cifar100",
#     train_transform=train_transform_resnet_cifar100,
#     val_transform=val_transform_resnet_cifar100,
#     batch_size=256
# )

# ## Finetuning ResNet 152 for CIFAR10

# resnet_cifar10 = ResNet(num_classes = 10)
# optimizer_resnet_cifar10 = torch.optim.AdamW(resnet_cifar10.parameters(), lr=1e-3, weight_decay=1e-4)

# train_classifier(
#     model = resnet_cifar10,
#     train_loader = train_loader_resnet_cifar10,
#     val_loader = val_loader_resnet_cifar10,
#     device = device,
#     optimizer = optimizer_resnet_cifar10,
#     epochs = 100,
#     use_amp = True,
#     name_model = "resnet152",
#     name_dataset="cifar10",
# )

# ## Finetuning ResNet 152 for CIFAR100
# resnet_cifar100 = ResNet(num_classes = 100)
# optimizer_resnet_cifar100 = torch.optim.AdamW(resnet_cifar100.parameters(), lr=1e-3, weight_decay=1e-4)

# train_classifier(
#     model = resnet_cifar100,
#     train_loader = train_loader_resnet_cifar100,
#     val_loader = val_loader_resnet_cifar100,
#     device = device,
#     optimizer = optimizer_resnet_cifar100,
#     epochs = 100,
#     use_amp = True,
#     name_model = "resnet152",
#     name_dataset="cifar100",
# )



# # Finetuning MLP-Mixer
# train_transform_mixer_cifar10 = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
# ])

# val_transform_mixer_cifar10 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
# ])

# train_transform_mixer_cifar100 = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
# ])

# val_transform_mixer_cifar100 = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
# ])


# train_loader_mixer_cifar10, val_loader_mixer_cifar10, _ = get_cifar_loaders(
#     dataset_name="cifar10",
#     train_transform=train_transform_mixer_cifar10,
#     val_transform=val_transform_mixer_cifar10,
#     batch_size=256
# )

# train_loader_mixer_cifar100, val_loader_mixer_cifar100, _ = get_cifar_loaders(
#     dataset_name="cifar100",
#     train_transform=train_transform_mixer_cifar100,
#     val_transform=val_transform_mixer_cifar100,
#     batch_size=256
# )


# ## Finetuning MLP-Mixer for CIFAR10
# mlp_mixer_cifar10 = MLPMixer(num_classes = 10)
# optimizer_mlpMixer_cifar10 = torch.optim.AdamW(mlp_mixer_cifar10.parameters(), lr=1e-3, weight_decay=1e-5)

# warmup_scheduler_mlpMixer_cifar10 = torch.optim.lr_scheduler.LinearLR(optimizer_mlpMixer_cifar10, start_factor=0.1, total_iters=5)
# cosine_scheduler_mlpMixer_cifar10 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mlpMixer_cifar10, T_max=300 - 5, eta_min=1e-6)

# scheduler_mlpMixer_cifar10 = torch.optim.lr_scheduler.SequentialLR(
#     optimizer_mlpMixer_cifar10, 
#     schedulers=[warmup_scheduler_mlpMixer_cifar10, cosine_scheduler_mlpMixer_cifar10],
#     milestones=[5]
# )


# train_classifier(
#     model = mlp_mixer_cifar10,
#     train_loader = train_loader_mixer_cifar10,
#     val_loader = val_loader_mixer_cifar10,
#     device = device,
#     optimizer = optimizer_mlpMixer_cifar10,
#     scheduler = scheduler_mlpMixer_cifar10,
#     epochs = 200,
#     use_amp = True,
#     name_model = "MLPMixer",
#     name_dataset="cifar10",
# )

# ## Finetuning MLP-Mixer for CIFAR100
# mlp_mixer_cifar100 = MLPMixer(num_classes = 100)

# optimizer_mlpMixer_cifar100 = torch.optim.AdamW(mlp_mixer_cifar100.parameters(), lr=1e-3, weight_decay=1e-5)

# warmup_scheduler_mlpMixer_cifar100 = torch.optim.lr_scheduler.LinearLR(optimizer_mlpMixer_cifar100, start_factor=0.1, total_iters=5)
# cosine_scheduler_mlpMixer_cifar100 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mlpMixer_cifar100, T_max=300 - 5, eta_min=1e-6)

# scheduler_mlpMixer_cifar100 = torch.optim.lr_scheduler.SequentialLR(
#     optimizer_mlpMixer_cifar100, 
#     schedulers=[warmup_scheduler_mlpMixer_cifar100, cosine_scheduler_mlpMixer_cifar100],
#     milestones=[5]
# )

# train_classifier(
#     model = mlp_mixer_cifar100,
#     train_loader = train_loader_mixer_cifar100,
#     val_loader = val_loader_mixer_cifar100,
#     device = device,
#     optimizer = optimizer_mlpMixer_cifar100,
#     scheduler = scheduler_mlpMixer_cifar100,
#     epochs = 200,
#     use_amp = True,
#     name_model = "MLPMixer",
#     name_dataset="cifar100",
# )


# # Finetuning ViT
# train_transform_vit_cifar = transforms.Compose([
#     transforms.Resize(256),
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandAugment(num_ops=2, magnitude=9),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# val_transform_vit_cifar = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
# ])

# train_loader_vit_cifar10, val_loader_vit_cifar10, _ = get_cifar_loaders(
#     dataset_name="cifar10",
#     train_transform=train_transform_vit_cifar,
#     val_transform=val_transform_vit_cifar,
#     batch_size=256
# )

# train_loader_vit_cifar100, val_loader_vit_cifar100, _ = get_cifar_loaders(
#     dataset_name="cifar100",
#     train_transform=train_transform_vit_cifar,
#     val_transform=val_transform_vit_cifar,
#     batch_size=256
# )


# ## Finetuning ViT for CIFAR10
# vit_cifar10 = ViT(num_classes=10)

# optimizer_vit_cifar10 = torch.optim.AdamW(vit_cifar10.parameters(), lr=3e-4, weight_decay=5e-2)

# warmup_cifar10 = torch.optim.lr_scheduler.LinearLR(optimizer_vit_cifar10, start_factor=0.1, total_iters=5)
# cosine_cifar10 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vit_cifar10, T_max=100-5, eta_min=1e-5)

# scheduler_vit_cifar10 = torch.optim.lr_scheduler.SequentialLR(
#     optimizer_vit_cifar10,
#     schedulers=[warmup_cifar10, cosine_cifar10],
#     milestones=[5]
# )

# train_classifier(
#     model = vit_cifar10,
#     train_loader = train_loader_vit_cifar10,
#     val_loader = val_loader_vit_cifar10,
#     device = device,
#     optimizer = optimizer_vit_cifar10,
#     scheduler = scheduler_vit_cifar10,
#     epochs = 100,
#     use_amp = True,
#     name_model = "ViT",
#     name_dataset="cifar10",
# )

# ## Finetuning ViT for CIFAR100
# vit_cifar100 = ViT(num_classes=100)

# optimizer_vit_cifar100 = torch.optim.AdamW(vit_cifar100.parameters(), lr=3e-4, weight_decay=5e-2)

# warmup_cifar100 = torch.optim.lr_scheduler.LinearLR(optimizer_vit_cifar100, start_factor=0.1, total_iters=5)
# cosine_cifar100 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_vit_cifar100, T_max=100-5, eta_min=1e-5)

# scheduler_vit_cifar100 = torch.optim.lr_scheduler.SequentialLR(
#     optimizer_vit_cifar100,
#     schedulers=[warmup_cifar100, cosine_cifar100],
#     milestones=[5]
# )

# train_classifier(
#     model = vit_cifar100,
#     train_loader = train_loader_vit_cifar100,
#     val_loader = val_loader_vit_cifar100,
#     device = device,
#     optimizer = optimizer_vit_cifar100,
#     scheduler = scheduler_vit_cifar100,
#     epochs = 100,
#     use_amp = True,
#     name_model = "ViT",
#     name_dataset="cifar100",
# )



# Finetuning MLP-Mixer pretrained on CIFAR10
train_transform_mixer_cifar10 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

val_transform_mixer_cifar10 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Finetuning MLP-Mixer pretrained on CIFAR100
train_transform_mixer_cifar100 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

val_transform_mixer_cifar100 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


train_loader_mixer_cifar10, val_loader_mixer_cifar10, _ = get_cifar_loaders(
    dataset_name="cifar10",
    train_transform=train_transform_mixer_cifar10,
    val_transform=val_transform_mixer_cifar10,
    batch_size=256
)

train_loader_mixer_cifar100, val_loader_mixer_cifar100, _ = get_cifar_loaders(
    dataset_name="cifar100",
    train_transform=train_transform_mixer_cifar100,
    val_transform=val_transform_mixer_cifar100,
    batch_size=256
)


## Finetuning MLP-Mixer for CIFAR10
mlp_mixer_cifar10 = MLPMixer(num_classes=10).to(device)

optimizer_mlpMixer_cifar10 = torch.optim.AdamW(
    mlp_mixer_cifar10.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

train_classifier(
    model=mlp_mixer_cifar10,
    train_loader=train_loader_mixer_cifar10,
    val_loader=val_loader_mixer_cifar10,
    device=device,
    optimizer=optimizer_mlpMixer_cifar10,
    epochs=30,
    use_amp=True,
    name_model="MLPMixer_b16_224",
    name_dataset="cifar10",
)


## Finetuning MLP-Mixer for CIFAR100
mlp_mixer_cifar100 = MLPMixer(num_classes=100).to(device)

optimizer_mlpMixer_cifar100 = torch.optim.AdamW(
    mlp_mixer_cifar100.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

train_classifier(
    model=mlp_mixer_cifar100,
    train_loader=train_loader_mixer_cifar100,
    val_loader=val_loader_mixer_cifar100,
    device=device,
    optimizer=optimizer_mlpMixer_cifar100,
    epochs=30,
    use_amp=True,
    name_model="MLPMixer_b16_224",
    name_dataset="cifar100",
)