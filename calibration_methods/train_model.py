import os
import time
import torch
from tqdm import tqdm
import torch.nn as nn
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)


@torch.no_grad()
def evaluate_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = logits.argmax(dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(total, 1)


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    scheduler = None,
    epochs=5,
    use_amp=True,
    name_model="model",
    name_dataset="dataset",
):

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    for epoch in tqdm(range(1, epochs + 1), desc=f"Epochs for {name_model}"):        
        model.train()
        running_loss = 0.0
        n_samples = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=(use_amp and device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            running_loss += loss.item() * bs
            n_samples += bs

        train_loss = running_loss / max(n_samples, 1)
        if epoch%5 == 0 :
            val_acc = evaluate_accuracy(model, val_loader, device)
            print(f"[{epoch}/{epochs}] loss={train_loss:.4f} val_acc={val_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/{name_model}_{name_dataset}.pt"
    torch.save(model.state_dict(), save_path)

    print(f"Model saved to: {save_path}")