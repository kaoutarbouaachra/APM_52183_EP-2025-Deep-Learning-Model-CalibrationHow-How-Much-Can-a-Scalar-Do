import argparse
import copy
import os
import pickle
import random
import sys

import numpy as np
import torch
import torchvision

sys.path.append(os.getcwd())

from mlp_mixer_pytorch import MLPMixer
from netcal.presentation import ReliabilityDiagram
from pathlib import Path
from torch.utils.data import DataLoader
from utils.data_utils import load_dataset, split_train_into_val
from utils.mixup_utils import GeneralMixupLoss
from utils.training_utils import full_train_test_loop

# This script trains and compares two models:
# a standard model (ERM) and a model trained with Mixup.
#
# 1. It configures hyperparameters (model, dataset, noise level, mixup_size) using argparse.
# 2. It prepares the data by creating three sets: Training, Validation (for Temperature Scaling), and Test.
# 3. It initializes two identical neural networks (e.g., ResNet18 or ResNeXt50).
# 4. It runs the full training loop:
#    - The ERM model is trained using standard CrossEntropy loss.
#    - The Mixup model is trained using mixed images (GeneralMixupLoss).
# 5. After training, Temperature Scaling is applied to both models.
# 6. It saves the results (logits, labels) and generates Reliability Diagrams
#    to visually compare calibration before and after correction.

# Set up command-line arguments.
parser = argparse.ArgumentParser(description="Hyperparameters for model training.")
parser.add_argument("--task-name", dest="task_name", default="MNIST", type=str)
parser.add_argument("--model-type", dest="model_type", default="ResNet18", type=str)
parser.add_argument("--optimizer", dest="optimizer", default="Adam", type=str)
parser.add_argument("--lr", dest="lr", default=1e-3, type=float)
parser.add_argument("--epochs", dest="epochs", default=200, type=int)
parser.add_argument("--num-runs", dest="n_runs", default=1, type=int)
parser.add_argument("--subsample", dest="subsample", default=0, type=int)
parser.add_argument("--label-noise", dest="label_noise", default=0, type=float)
parser.add_argument("--val-prop", dest="val_prop", default=0.1, type=float)
parser.add_argument("--mix-size", dest="mix_size", default=2, type=int)
parser.add_argument("--save-model", dest="save_model", action="store_true")
parser.set_defaults(save_model=False)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device != "cpu":
    print("Device count: ", torch.cuda.device_count())
    print("GPU being used: {}".format(torch.cuda.get_device_name(0)))

# Fix seeds for reproducibility.
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# Set up runs path.
runs_path = f"runs/{args.task_name}_{args.model_type}_{args.optimizer}_{args.epochs}_epochs_{args.n_runs}_runs_{args.subsample}_subsample_{args.mix_size}_mix_{args.label_noise}_noise"
Path(runs_path).mkdir(parents=True, exist_ok=True)
perf_file = open(f"{runs_path}/training.out", "w")

# Set up transform parameters depending on model.
rescale, custom_normalizer = None, None
if args.model_type == "ViT":
    rescale = 224  # Standard rescaling size.

# Load data.
train_data, test_data, n_channels, out_dim = load_dataset(
    dataset=args.task_name,
    rescale=rescale,
    custom_normalizer=custom_normalizer,
    subsample=args.subsample,
    label_noise=args.label_noise,
)

# Model/training parameters.
mixup_alpha = 1
lr = args.lr
epochs = args.epochs
batch_size = 500
num_runs = args.n_runs
n_bins = 15

print(
    "The following model/training parameters were used for this run: \n", file=perf_file
)
print("val_prop = ", args.val_prop, file=perf_file)
print("batch_size = ", batch_size, file=perf_file)
print("mixup_alpha = ", mixup_alpha, file=perf_file)
print("mix_size = ", args.mix_size, file=perf_file)
print("lr = ", lr, file=perf_file)
print("epochs = ", epochs, file=perf_file)
print("num_runs = ", num_runs, file=perf_file)
print("n_bins = ", n_bins, file=perf_file)
print("-------------------------------------------------\n", file=perf_file)

# Prepare data.
train_data, val_data = split_train_into_val(train_data, args.val_prop)
train_dl = DataLoader(train_data, batch_size=batch_size, shuffle=True)
cal_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dl = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Set up models.
# None of these models work when there is only one color channel.
if args.model_type == "ResNet18":  # Small model
    base_model = torchvision.models.resnet18(num_classes=out_dim).to(device)
elif args.model_type == "ResNeXt50":
    base_model = torchvision.models.resnext50_32x4d(num_classes=out_dim).to(device)
elif args.model_type == "DenseNet":
    base_model = torchvision.models.densenet121(
        num_classes=out_dim, memory_efficient=False
    ).to(device)
elif args.model_type == "ConvNeXt":
    base_model = torchvision.models.convnext_tiny(num_classes=out_dim).to(device)
elif args.model_type == "MLPMixer":
    base_model = MLPMixer(
        image_size=32,
        channels=n_channels,
        patch_size=8,
        dim=1024,
        depth=8,
        num_classes=out_dim,
    ).to(device)
elif args.model_type == "MobileNetV2":  # Small model
    base_model = torchvision.models.mobilenet_v2(num_classes=out_dim).to(device)
else:
    sys.exit(f"{args.model_type} is an unsupported model type.")

mixup_model = copy.deepcopy(base_model)

criterion = torch.nn.CrossEntropyLoss()
mixup_criterion = GeneralMixupLoss(
    torch.nn.CrossEntropyLoss(reduction="none"),
    alpha=mixup_alpha,
    mix_size=args.mix_size,
)

if args.optimizer == "SGD":
    base_optimizer = torch.optim.SGD(base_model.parameters(), lr=lr)
    mixup_optimizer = torch.optim.SGD(mixup_model.parameters(), lr=lr)
else:
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=lr)
    mixup_optimizer = torch.optim.Adam(mixup_model.parameters(), lr=lr)

# Train Mixup and base models.
_, _, mixup_confidences, mixup_ts_confidences, mixup_labels = full_train_test_loop(
    model=mixup_model,
    test_loader=test_dl,
    test_loss_fn=criterion,
    train_loader=train_dl,
    train_loss_fn=mixup_criterion,
    cal_loader=cal_dl,
    optimizer=mixup_optimizer,
    num_epochs=epochs,
    batch_size=batch_size,
    model_name="Mixup",
    out_file=perf_file,
    num_runs=num_runs,
    log_epoch_stats=False,
    n_bins=n_bins,
    device=device,
)

train_logits, train_labels, erm_confidences, erm_ts_confidences, erm_labels = full_train_test_loop(
    model=base_model,
    test_loader=test_dl,
    test_loss_fn=criterion,
    train_loader=train_dl,
    train_loss_fn=criterion,
    cal_loader=cal_dl,
    optimizer=base_optimizer,
    num_epochs=epochs,
    batch_size=batch_size,
    model_name="ERM",
    out_file=perf_file,
    num_runs=num_runs,
    log_epoch_stats=False,
    n_bins=n_bins,
    device=device,
)

# Save models if requested.
if args.save_model:
    pickle.dump(mixup_model, open(f"{runs_path}/mixup_model_gpu.p", "wb"))
    pickle.dump(mixup_model.to(torch.device("cpu")), open(f"{runs_path}/mixup_model_cpu.p", "wb"))
    pickle.dump(base_model, open(f"{runs_path}/erm_model_gpu.p", "wb"))
    pickle.dump(base_model.to(torch.device("cpu")), open(f"{runs_path}/erm_model_cpu.p", "wb"))

# Save logits, confidences, and labels.
pickle.dump(mixup_confidences, open(f"{runs_path}/mixup_confidences.p", "wb"))
pickle.dump(mixup_ts_confidences, open(f"{runs_path}/mixup_ts_confidences.p", "wb"))
pickle.dump(mixup_labels, open(f"{runs_path}/mixup_labels.p", "wb"))

pickle.dump(train_logits, open(f"{runs_path}/erm_train_logits.p", "wb"))
pickle.dump(train_labels, open(f"{runs_path}/erm_train_labels.p", "wb"))
pickle.dump(erm_confidences, open(f"{runs_path}/erm_confidences.p", "wb"))
pickle.dump(erm_ts_confidences, open(f"{runs_path}/erm_ts_confidences.p", "wb"))
pickle.dump(erm_labels, open(f"{runs_path}/erm_labels.p", "wb"))

# Generate reliability diagrams.
diagram = ReliabilityDiagram(n_bins)

diagram.plot(mixup_confidences, mixup_labels, filename=f"{runs_path}/mixup_rel_diagram.png")
diagram.plot(mixup_ts_confidences, mixup_labels, filename=f"{runs_path}/mixup_ts_rel_diagram.png")

diagram.plot(erm_confidences, erm_labels, filename=f"{runs_path}/erm_rel_diagram.png")
diagram.plot(erm_ts_confidences, erm_labels, filename=f"{runs_path}/erm_ts_rel_diagram.png")

