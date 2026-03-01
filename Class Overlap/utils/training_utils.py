import numpy as np
import torch
from models.temperature_scaled_model import TemperatureScaledModel
from netcal.metrics import ACE, ECE, MCE
from utils.mixup_utils import GeneralMixupLoss


def reset_weights(m):
    """
    Reinitialize the parameters of a module if it implements
    a `reset_parameters()` method.

    This is used to restart training from scratch
    across multiple experimental runs.
    """
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_grad_norm(model):
    """
    Compute the global L2 norm of all model gradients.

    Useful for diagnosing:
    - exploding gradients
    - vanishing gradients
    """
    grad_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item() ** 2

    return grad_norm ** 0.5


def get_model_param_tensor(model):
    """
    Flatten and concatenate all model parameters
    into a single 1D tensor.

    This is useful for mathematical analysis
    or comparing parameter states across runs.
    """
    flattened_params = []

    for param_tensor in model.parameters():
        flattened_params.append(torch.flatten(param_tensor))

    return torch.cat(flattened_params)


def get_model_evaluations(model, data_loader, device="cpu"):
    """
    Switch the model to evaluation mode and return
    the Softmax probabilities computed on the last batch
    of the provided data loader.
    """
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    output = None

    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            output = softmax(model(data))

    return output


def train(
    model,
    train_loader,
    loss_fn,
    optimizer,
    epoch,
    batch_size,
    out_file,
    log_epoch_stats=False,
    device="cpu",
):
    """
    Perform one full training epoch.

    Supports:
    - Standard ERM training
    - Mixup training (if loss_fn is GeneralMixupLoss)
    """
    model.train()

    use_mixup = isinstance(loss_fn, GeneralMixupLoss)
    avg_batch_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):

        # Move data to device (CPU or GPU)
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass + loss computation
        if use_mixup:
            loss = loss_fn(model, data, target)
        else:
            output = model(data)
            loss = loss_fn(output, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Accumulate average loss over the epoch
        avg_batch_loss += loss.item() / len(train_loader)

    if log_epoch_stats:
        print(
            f"[Epoch {epoch}] Average Batch Loss: {avg_batch_loss}",
            file=out_file,
        )


def test(model, test_loader, loss_fn, out_file, device="cpu"):
    """
    Compute classification error (percentage error).

    Returns: 100 * (1 - accuracy)
    """
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            correct += pred.eq(target.view_as(pred)).sum().item()

    return 100 * (1 - (correct / len(test_loader.dataset)))


def compute_nll(model, test_loader, device="cpu"):
    """
    Compute the average Negative Log-Likelihood (NLL).

    This is the primary probabilistic performance
    metric used in the paper.
    """
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    avg_loss = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            avg_loss += (
                criterion(output, target).item()
                / len(test_loader.dataset)
            )

    return avg_loss


def get_confidences_and_labels(model, test_loader, device="cpu"):
    """
    Return:
    - Raw logits
    - Softmax probabilities
    - True labels

    Required for computing calibration metrics.
    """
    logits = []
    softmaxes = []
    labels = []

    with torch.no_grad():
        for data, target in test_loader:
            out = model(data.to(device))

            logits.append(out)
            softmaxes.append(torch.nn.functional.softmax(out, dim=1))
            labels.append(target)

    logits = torch.cat(logits)
    softmaxes = torch.cat(softmaxes)
    labels = torch.cat(labels)

    return (
        logits.cpu().numpy(),
        softmaxes.cpu().numpy(),
        labels.cpu().numpy(),
    )


def get_confidence_metrics(confidences, labels, n_bins=15):
    """
    Compute calibration metrics:

    - ACE: Adaptive Calibration Error
    - ECE: Expected Calibration Error
    - MCE: Maximum Calibration Error
    """
    ace = ACE(n_bins)
    ece = ECE(n_bins)
    mce = MCE(n_bins)

    return [
        ace.measure(confidences, labels),
        ece.measure(confidences, labels),
        mce.measure(confidences, labels),
    ]


def full_train_test_loop(
    model,
    test_loader,
    test_loss_fn,
    train_loader,
    train_loss_fn,
    cal_loader,
    optimizer,
    num_epochs,
    batch_size,
    model_name,
    out_file,
    num_runs=10,
    log_epoch_stats=False,
    n_bins=15,
    device="cpu",
):
    """
    COMPLETE EXPERIMENTAL LOOP:

    For each run:
    1. Reset model weights
    2. Train for num_epochs
    3. Evaluate baseline performance
    4. Apply Temperature Scaling
    5. Evaluate calibrated performance

    Also returns training logits for further analysis.
    """

    cal_metrics = []
    ts_cal_metrics = []
    nll = []
    ts_nll = []

    print("{} model results: ".format(model_name), file=out_file)

    for i in range(num_runs):

        # Reset model parameters
        model.apply(reset_weights)

        # Training phase
        for j in range(1, num_epochs + 1):
            train(
                model,
                train_loader,
                train_loss_fn,
                optimizer,
                j,
                batch_size,
                out_file,
                log_epoch_stats,
                device,
            )

        # ----- Evaluation BEFORE temperature scaling -----
        _, confidences, labels = get_confidences_and_labels(
            model, test_loader, device
        )

        cal_metrics.append(
            get_confidence_metrics(confidences, labels, n_bins)
        )

        nll.append(compute_nll(model, test_loader, device))

        # ----- Temperature Scaling -----
        ts_model = TemperatureScaledModel(model, device=device)
        ts_model.fit(cal_loader)

        # ----- Evaluation AFTER temperature scaling -----
        _, ts_confidences, _ = get_confidences_and_labels(
            ts_model, test_loader, device
        )

        ts_cal_metrics.append(
            get_confidence_metrics(ts_confidences, labels, n_bins)
        )

        ts_nll.append(compute_nll(ts_model, test_loader, device))

    # Training logits for theoretical analysis
    train_logits, _, train_labels = get_confidences_and_labels(
        model, train_loader, device
    )

    # Convert lists to numpy arrays
    cal_metrics = np.array(cal_metrics)
    ts_cal_metrics = np.array(ts_cal_metrics)
    nll = np.array(nll)
    ts_nll = np.array(ts_nll)

    # Final train/test errors
    train_error = test(model, train_loader, test_loss_fn, out_file, device)
    test_error = test(model, test_loader, test_loss_fn, out_file, device)

    print(f"Last Train Error: {train_error:.4f}", file=out_file)
    print(f"Last Test Error: {test_error:.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    # Calibration before TS
    print(f"Average ACE: {cal_metrics[:, 0].mean():.4f}\t 1 std: {cal_metrics[:, 0].std():.4f}", file=out_file)
    print(f"Average ECE: {cal_metrics[:, 1].mean():.4f}\t 1 std: {cal_metrics[:, 1].std():.4f}", file=out_file)
    print(f"Average MCE: {cal_metrics[:, 2].mean():.4f}\t 1 std: {cal_metrics[:, 2].std():.4f}", file=out_file)
    print(f"Average NLL: {nll.mean():.4f}\t 1 std: {nll.std():.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    # Calibration after TS
    print(f"Average Post-TS ACE: {ts_cal_metrics[:, 0].mean():.4f}\t 1 std: {ts_cal_metrics[:, 0].std():.4f}", file=out_file)
    print(f"Average Post-TS ECE: {ts_cal_metrics[:, 1].mean():.4f}\t 1 std: {ts_cal_metrics[:, 1].std():.4f}", file=out_file)
    print(f"Average Post-TS MCE: {ts_cal_metrics[:, 2].mean():.4f}\t 1 std: {ts_cal_metrics[:, 2].std():.4f}", file=out_file)
    print(f"Average Post-TS NLL: {ts_nll.mean():.4f}\t 1 std: {ts_nll.std():.4f}", file=out_file)
    print("-------------------------------------------------\n", file=out_file)

    return train_logits, train_labels, confidences, ts_confidences, labels
