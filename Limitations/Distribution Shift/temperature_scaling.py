import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

# =============================================================================
# 1. ABSTRACT BASE CLASS
# =============================================================================

class PostHocCalibrator(nn.Module):
    """
    Abstract base class for all post-hoc calibration methods.
    Enforces a standard interface for 'fit' and 'predict'.
    """
    def __init__(self):
        super().__init__()

    def fit(self, logits, labels):
        """
        Learn calibration parameters (T, weights, etc.) on a validation set.
        
        Args:
            logits (torch.Tensor): Validation logits (N, C)
            labels (torch.Tensor): Validation labels (N,)
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, logits):
        """
        Apply calibration to test logits.
        
        Args:
            logits (torch.Tensor): Test logits (N, C)
            
        Returns:
            probs (torch.Tensor): Calibrated probabilities (N, C) summing to 1.
        """
        raise NotImplementedError("Subclasses must implement predict()")


# =============================================================================
# 2. TEMPERATURE SCALING (Refactored)
# =============================================================================

class TemperatureScaling(PostHocCalibrator):
    """
    Standard Temperature Scaling (Guo et al. 2017).
    Optimizes a single scalar T to minimize NLL on validation data.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def fit(self, logits, labels):
        """
        Tunes the temperature parameter using L-BFGS optimization on NLL.
        """
        self.cuda()
        logits = logits.cuda()
        labels = labels.cuda()
        
        nll_criterion = nn.CrossEntropyLoss().cuda()
        
        # Optimizer
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        
        # Clamp ensures T > 0 to avoid numerical instability
        with torch.no_grad():
             self.temperature.clamp_(min=1e-2)

        print(f"Optimal Temperature found: {self.temperature.item():.4f}")
        return self

    def predict(self, logits):
        """
        Returns Softmax(logits / T).
        """
        scaled_logits = self.temperature_scale(logits)
        return F.softmax(scaled_logits, dim=1)

    def temperature_scale(self, logits):
        """
        Helper to just scale logits (useful for metrics expecting logits).
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


# =============================================================================
# 3. COMPATIBILITY WRAPPER (Keeps your old scripts working)
# =============================================================================

class ModelWithTemperature(nn.Module):
    """
    A wrapper that bundles a base model with a calibrator.
    This replaces your old class to ensure 'evaluate_shift.py' still works.
    """
    def __init__(self, model, method='ts', num_classes=100):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        
        if method == 'ts':
            self.calibrator = TemperatureScaling()
        elif method == 'ets':
            self.calibrator = EnsembleTemperatureScaling(num_classes)
        elif method == 'tva':
            self.calibrator = TopVersusAll(num_classes)
        elif method == 'dac':
            self.calibrator = DensityAwareCalibration()
        else:
            raise ValueError(f"Unknown calibration method: {method}")


    def forward(self, input):
        """
        Standard forward pass returns LOGITS (scaled if calibrated).
        We return logits because your evaluation scripts expect them for CrossEntropyLoss.
        """
        logits = self.model(input)
        
        # If it's TemperatureScaling, we can return scaled logits directly.
        # This maintains compatibility with your existing metrics that need logits.
        if isinstance(self.calibrator, TemperatureScaling):
            return self.calibrator.temperature_scale(logits)
        
        # For other methods (TvA, ETS) that return probabilities, 
        # we might need to return log(probs) to approximate logits.
        probs = self.calibrator.predict(logits)
        return torch.log(probs + 1e-12) # Log-probs for stability

    def set_temperature(self, valid_loader):
        """
        Collects data from loader and calls fit().
        Maintains the API expected by your existing scripts.
        """
        self.model.eval()
        logits_list = []
        labels_list = []
        
        print("Collecting validation logits...")
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Fit the internal calibrator
        self.calibrator.fit(logits, labels)
        
        # Check performance
        ece_criterion = ECELoss()
        nll_criterion = nn.CrossEntropyLoss()
        
        # Note: We use self.forward() here which uses the calibrated output
        # But we need to pass dummy input, so we reuse the cached logits logic roughly
        # Actually, let's just use the calibrator directly for the report
        if isinstance(self.calibrator, TemperatureScaling):
            after_logits = self.calibrator.temperature_scale(logits)
            nll = nll_criterion(after_logits, labels).item()
            ece = ece_criterion(after_logits, labels).item()
            print(f'After calibration - NLL: {nll:.3f}, ECE: {ece:.3f}')
            
        return self
    
    @property
    def temperature(self):
        # Expose temperature param for your scripts that print it
        if hasattr(self.calibrator, 'temperature'):
            return self.calibrator.temperature
        return torch.tensor(1.0) # Dummy for other methods


# =============================================================================
# 4. METRICS (Old + New Adaptive ECE)
# =============================================================================

class ECELoss(nn.Module):
    """
    Standard Expected Calibration Error (Fixed Bins).
    Ovadia et al. (2019) Benchmark Standard.
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


class AdaptiveECELoss(nn.Module):
    """
    Adaptive Expected Calibration Error.
    Uses quantiles to ensure every bin has the same number of samples.
    Recommended by modern papers (e.g. DAC, TvA) to reduce bias.
    """
    def __init__(self, n_bins=15):
        super(AdaptiveECELoss, self).__init__()
        self.n_bins = n_bins

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        
        n = confidences.shape[0]
        # Sort by confidence
        confidences, sorted_idx = torch.sort(confidences)
        accuracies = accuracies[sorted_idx]

        ece = torch.zeros(1, device=logits.device)
        bin_size = n // self.n_bins
        
        for i in range(self.n_bins):
            # Last bin gets the remainder
            start = i * bin_size
            end = (i + 1) * bin_size if i < self.n_bins - 1 else n
            
            if start >= end: break
            
            bin_conf = confidences[start:end]
            bin_acc = accuracies[start:end].float()
            
            abs_diff = torch.abs(bin_conf.mean() - bin_acc.mean())
            ece += (end - start) / n * abs_diff
            
        return ece

# Wrapper for backward compatibility if you import _ECELoss elsewhere
_ECELoss = ECELoss


#############################################################################
#############################################################################

class EnsembleTemperatureScaling(PostHocCalibrator):
    """
    Ensemble Temperature Scaling (ETS) from Zhang et al. (2020).
    Learns a weighted ensemble of:
    1. Temperature Scaled Logits
    2. Original Logits (Identity)
    3. Uniform Distribution
    
    Prob = w1 * Softmax(logits/T) + w2 * Softmax(logits) + w3 * (1/K)
    """
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        # T parameter (initialize to 1.5 like standard TS)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        # Weights for the 3 components (unconstrained raw scores)
        self.w_raw = nn.Parameter(torch.zeros(3)) 

    def fit(self, logits, labels):
        self.cuda()
        logits = logits.cuda()
        labels = labels.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        
        # Optimize T and weights jointly
        optimizer = optim.LBFGS([self.temperature, self.w_raw], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            # Get weights via softmax to ensure they sum to 1
            w = F.softmax(self.w_raw, dim=0)
            
            # 1. Temperature Scaled
            t_logits = logits / self.temperature.unsqueeze(1).expand_as(logits)
            p1 = F.softmax(t_logits, dim=1)
            
            # 2. Original
            p2 = F.softmax(logits, dim=1)
            
            # 3. Uniform
            p3 = torch.ones_like(p1) / self.num_classes
            
            # Weighted Sum
            p_final = w[0] * p1 + w[1] * p2 + w[2] * p3
            
            # NLL Loss (Manual calculation since we have probs, not logits)
            # Clip for numerical stability
            p_final = p_final.clamp(min=1e-12)
            loss = torch.mean(-torch.log(p_final[range(len(labels)), labels]))
            
            loss.backward()
            return loss

        optimizer.step(eval_step)
        
        # Clamp T > 0
        with torch.no_grad():
             self.temperature.clamp_(min=1e-2)
             
        w = F.softmax(self.w_raw, dim=0).detach().cpu().numpy()
        print(f"ETS Fitted: T={self.temperature.item():.3f}, Weights={w}")
        return self

    def predict(self, logits):
        w = F.softmax(self.w_raw, dim=0)
        
        # 1. Temperature Scaled
        t = self.temperature.unsqueeze(1).expand_as(logits)
        p1 = F.softmax(logits / t, dim=1)
        
        # 2. Original
        p2 = F.softmax(logits, dim=1)
        
        # 3. Uniform
        p3 = torch.ones_like(p1) / self.num_classes
        
        return w[0] * p1 + w[1] * p2 + w[2] * p3


class TopVersusAll(PostHocCalibrator):
    """
    Top-versus-All (TvA) Calibration (2024).
    Treats multi-class calibration as K one-vs-rest binary problems.
    Learns a separate logistic regression (Platt scaler) for each class.
    """
    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        # Learn a and b for each class: sigmoid(a * logit + b)
        self.a = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def fit(self, logits, labels):
        self.cuda()
        logits = logits.cuda()
        labels = labels.cuda()
        
        # We optimize NLL for each class independently
        # But for speed in PyTorch, we can do it in a vectorized way
        # or just a simple loop if classes are < 1000
        
        optimizer = optim.LBFGS([self.a, self.b], lr=0.01, max_iter=50)
        
        # Create binary targets for each class: 1 if label==k, else 0
        # shape: (N, K)
        y_onehot = F.one_hot(labels, num_classes=self.num_classes).float()

        def eval_step():
            optimizer.zero_grad()
            
            # Vectorized Logistic Regression
            # calibrated_logit_k = a_k * logit_k + b_k
            cal_logits = self.a.unsqueeze(0) * logits + self.b.unsqueeze(0)
            
            # Binary Cross Entropy with Logits
            # We want to minimize BCE for each class and sum them up
            loss = F.binary_cross_entropy_with_logits(cal_logits, y_onehot)
            
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print("TvA Fitted.")
        return self

    def predict(self, logits):
        # Apply binary calibration to every logit
        cal_logits = self.a.unsqueeze(0) * logits + self.b.unsqueeze(0)
        
        # Convert to probabilities using Sigmoid (since they are binary probs)
        probs = torch.sigmoid(cal_logits)
        
        # Normalize so they sum to 1 (Standard TvA step)
        probs = probs / probs.sum(dim=1, keepdim=True)
        return probs


class DensityAwareCalibration(PostHocCalibrator):
    """
    Density-Aware Calibration (DAC) - Logit-Distance Version.
    Scales temperature based on distance to training data in logit space.
    T(x) = T_base + alpha * Distance(x)
    """
    def __init__(self, num_bins=10):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.alpha = nn.Parameter(torch.zeros(1)) # Sensitivity to OOD
        self.prototypes = None # Will store cluster centers or mean logits

    def fit(self, logits, labels):
        self.cuda()
        logits = logits.cuda()
        labels = labels.cuda()
        
        # 1. Density Estimation Setup
        # Simple version: Compute mean logit vector for correct predictions
        # A more complex version uses kNN on all training features
        with torch.no_grad():
            self.train_mean = logits.mean(dim=0, keepdim=True)
            self.train_std = logits.std(dim=0, keepdim=True) + 1e-6
        
        # Optimize T and Alpha
        optimizer = optim.LBFGS([self.temperature, self.alpha], lr=0.01, max_iter=50)

        def eval_step():
            optimizer.zero_grad()
            
            # Calculate Density (Distance)
            # Normalized Euclidean distance to the global mean of validation set
            # (proxy for "how weird is this sample")
            z_score = (logits - self.train_mean) / self.train_std
            dist = z_score.norm(dim=1)
            
            # Adaptive Temperature
            # If dist is high (OOD), T increases -> confidence drops
            # Softplus ensures T > 0
            t_adaptive = F.softplus(self.temperature + self.alpha * dist).unsqueeze(1)
            
            loss = nn.CrossEntropyLoss()(logits / t_adaptive, labels)
            loss.backward()
            return loss

        optimizer.step(eval_step)
        print(f"DAC Fitted: Base T={self.temperature.item():.3f}, Alpha={self.alpha.item():.3f}")
        return self

    def predict(self, logits):
        # 1. Compute Distance
        z_score = (logits - self.train_mean) / self.train_std
        dist = z_score.norm(dim=1)
        
        # 2. Compute Adaptive T
        t_adaptive = F.softplus(self.temperature + self.alpha * dist).unsqueeze(1)
        
        return F.softmax(logits / t_adaptive, dim=1)

