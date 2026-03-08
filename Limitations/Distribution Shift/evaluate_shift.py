import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from models import DenseNet
from temperature_scaling import ModelWithTemperature, ECELoss, AdaptiveECELoss
from corruptions import CorruptedDataset, CORRUPTION_DICT


# ── Reliability bins (numpy arrays, no torch dependency) ────────────────────
def compute_reliability_bins(confidences: np.ndarray,
                              labels: np.ndarray,
                              preds: np.ndarray,
                              n_bins: int = 15) -> list:
    bin_edges =np.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi  = bin_edges[i], bin_edges[i + 1]
        mask= (confidences >= lo) & (confidences < hi)
        n       = int(mask.sum())
        if n > 0:
            bins.append({
                'conf': float(confidences[mask].mean()),
                'acc':  float((preds[mask] == labels[mask]).mean()),
                'n':    n
            })
        else:
            bins.append({
                'conf': float((lo + hi) / 2),
                'acc':  None,
                'n':    0
            })
    return bins


# ── Metrics (returns dict + raw numpy arrays for bin computation) ────────────
def get_metrics(logits: torch.Tensor,
                labels: torch.Tensor,
                n_bins: int = 15) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (metrics_dict, confidences_np, labels_np, preds_np).
    Keeping raw arrays avoids a second forward pass for bin computation.
    """
    labels = labels.to(logits.device)

    softmaxes             = torch.softmax(logits, dim=1)
    confidences, preds    = torch.max(softmaxes, dim=1)

    acc = preds.eq(labels).float().mean().item()
    nll = torch.nn.CrossEntropyLoss()(logits, labels).item()
    ece = ECELoss(n_bins)(logits, labels).item()

    ada_raw = AdaptiveECELoss(n_bins)(logits, labels)
    ada_ece = ada_raw.item() if torch.is_tensor(ada_raw) else float(ada_raw)

    metrics = {
        "accuracy":     acc,
        "nll":          nll,
        "ece":          ece,
        "adaptive_ece": ada_ece,
    }

    conf_np  = confidences.cpu().numpy()
    labels_np =labels.cpu().numpy()
    preds_np= preds.cpu().numpy()

    return metrics, conf_np, labels_np, preds_np


# ── Thin dataset wrapper (defined once, not inside the loop) ─────────────────
class NormalizedDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]
        return self.transform(img), target


# ── Main evaluation loop ─────────────────────────────────────────────────────
def evaluate_shift(data_dir: str,
                   save_dir: str,
                   batch_size: int = 256,
                   depth: int = 40,
                   growth_rate: int = 12):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path   = os.path.join(save_dir, 'model.pth')
    indices_path = os.path.join(save_dir, 'valid_indices.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model at {model_path}. Run train.py first.")

    block_config =[(depth - 4) // 6] * 3
    orig_model= DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=100
    ).to(device)
    orig_model.load_state_dict(torch.load(model_path, map_location=device))
    orig_model.eval()

    # ── Transforms & datasets ────────────────────────────────────────────────
    mean = [0.5071, 0.4867, 0.4408]
    std  =[0.2675, 0.2565, 0.2761]
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if os.path.exists(indices_path):
        valid_indices = torch.load(indices_path)
    else:
        print("Warning: valid_indices.pth not found — using random split.")
        valid_indices = torch.randperm(50000)[:5000]

    train_set    = dset.CIFAR100(data_dir, train=True,  download=False, transform=normalize)
    test_norm    = dset.CIFAR100(data_dir, train=False, download=False, transform=normalize)
    test_raw     = dset.CIFAR100(data_dir, train=False, download=False, transform=None)

    valid_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=SubsetRandomSampler(valid_indices),
                              pin_memory=True, num_workers=2)
    clean_loader = DataLoader(test_norm,  batch_size=batch_size,
                              shuffle=False, pin_memory=True, num_workers=2)

    corruption_names = list(CORRUPTION_DICT.keys())

    # ── Shared inference helper ───────────────────────────────────────────────
    def run_inference(model, loader):
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                logits_list.append(model(inputs.to(device)))
                labels_list.append(targets)
        return torch.cat(logits_list), torch.cat(labels_list).to(device)

    # ── Method loop ───────────────────────────────────────────────────────────
    all_results = {}

    for method in ['ts', 'ets', 'tva', 'dac']:
        print(f"\n{'='*42}\n  Method: {method.upper()}\n{'='*42}")

        cal_model= ModelWithTemperature(orig_model, method=method, num_classes=100)
        cal_model.set_temperature(valid_loader)

        method_results = {"corruptions": {}}

        # ── Clean baseline ────────────────────────────────────────────────────
        logits, labels =run_inference(cal_model, clean_loader)
        metrics, conf_np, labels_np, preds_np =get_metrics(logits, labels)
        metrics['bins'] = compute_reliability_bins(conf_np, labels_np, preds_np)
        method_results['clean_baseline'] = metrics
        print(f"  Clean → Acc={metrics['accuracy']:.4f}  "
              f"ECE={metrics['ece']:.4f}  AdaECE={metrics['adaptive_ece']:.4f}")

        # ── Corruption loop ───────────────────────────────────────────────────
        for name in tqdm(corruption_names, desc=f"{method.upper()} corruptions"):
            method_results['corruptions'][name] = {}

            for severity in range(1, 6):
                corrupted_ds  = CorruptedDataset(test_raw,
                                                 corruption_name=name,
                                                 severity=severity)
                final_ds= NormalizedDataset(corrupted_ds, normalize)
                loader        = DataLoader(final_ds, batch_size=batch_size,
                                           shuffle=False, num_workers=2,
                                           pin_memory=True)

                logits, labels                       = run_inference(cal_model, loader)
                metrics, conf_np, labels_np, preds_np = get_metrics(logits, labels)
                metrics['bins']                      = compute_reliability_bins(
                                                           conf_np, labels_np, preds_np)
                method_results['corruptions'][name][severity] =metrics

            avg_ece = np.mean([
                method_results['corruptions'][name][s]['ece']
                for s in range(1, 6)
            ])
            tqdm.write(f"  {name:<22} avg ECE = {avg_ece:.4f}")

        all_results[method] = method_results

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(save_dir, 'ovadia_plus_results.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved → {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--save', type=str, default='./checkpoints')
    args = parser.parse_args()
    evaluate_shift(args.data, args.save)
