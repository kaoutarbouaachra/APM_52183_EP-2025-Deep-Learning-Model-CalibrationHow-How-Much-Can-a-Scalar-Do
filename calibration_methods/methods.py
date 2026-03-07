import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from sklearn.isotonic import IsotonicRegression 

# Histogramm Binning
class HistogrammBinningCalibrator:
    def __init__(self, bins):
        self.eps = 1e-12

        self.bins = bins
        self.hb = None              
        self.prior = None           
        self.num_classes = None
        self.is_fitted = False
        self.device = None

    def fit(self, probs_val: torch.Tensor, y_val: torch.Tensor):
        """
        Args:
            probs_val: Tensor (N, K), probabilités
            y_val: Tensor (N,), labels entiers dans [0, K-1]
        """
        if probs_val.ndim != 2:
            raise ValueError("probs_val doit être 2D (N, K)")
        if y_val.ndim != 1:
            raise ValueError("y_val doit être 1D (N,)")
        if probs_val.shape[0] != y_val.shape[0]:
            raise ValueError("probs_val et y_val doivent avoir le même nombre d'exemples")

        N, K = probs_val.shape
        self.device = probs_val.device
        dtype = probs_val.dtype
        device = probs_val.device

        y_long = y_val.to(device=device, dtype=torch.long)

        if y_long.min() < 0 or y_long.max() >= K:
            raise ValueError("y_val contient des labels hors [0, K-1]")

        edges = torch.as_tensor(self.bins, dtype=dtype, device=device)
        if edges.ndim != 1 or edges.numel() < 2:
            raise ValueError("bins doit contenir au moins 2 bords")
        if not torch.all(edges[1:] >= edges[:-1]):
            raise ValueError("bins doit être trié dans l'ordre croissant")

        M = edges.numel() - 1
        inner_edges = edges[1:-1]

        # prior par classe
        class_counts = torch.bincount(y_long, minlength=K).to(dtype=dtype)
        class_prior = class_counts / max(N, 1)

        hb = []

        for k in range(K):
            scores_k = probs_val[:, k].clamp(0.0, 1.0)

            if inner_edges.numel() == 0:
                bin_idx = torch.zeros(N, dtype=torch.long, device=device)
            else:
                bin_idx = torch.bucketize(scores_k, inner_edges, right=False)
                bin_idx = bin_idx.clamp(0, M - 1)

            bin_counts = torch.bincount(bin_idx, minlength=M).to(dtype=dtype)

            yk = (y_long == k).to(dtype=dtype)
            pos_counts = torch.bincount(bin_idx, weights=yk, minlength=M).to(dtype=dtype)
            theta_k = torch.empty(M, dtype=dtype, device=device)

            nonempty = bin_counts > 0
            theta_k[nonempty] = pos_counts[nonempty] / bin_counts[nonempty]

            theta_k[~nonempty] = class_prior[k]

            hb.append(theta_k)

        self.hb = hb
        self.prior = class_prior
        self.num_classes = K
        self.is_fitted = True
        return self

    def calibrate(self, probs_test: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted or self.hb is None or self.prior is None or self.num_classes is None:
            raise RuntimeError("Call fit() before calibrate().")

        if probs_test.ndim != 2:
            raise ValueError("probs_test doit être 2D (N, K)")

        N, K = probs_test.shape
        if K != self.num_classes:
            raise ValueError(f"probs_test a K={K}, attendu {self.num_classes}")

        device = probs_test.device
        dtype = probs_test.dtype

        edges = torch.as_tensor(self.bins, dtype=dtype, device=device)
        M = edges.numel() - 1
        inner_edges = edges[1:-1]

        calibrated = torch.empty_like(probs_test)

        for k in range(K):
            scores_k = probs_test[:, k].clamp(0.0, 1.0)

            if inner_edges.numel() == 0:
                bin_idx = torch.zeros(N, dtype=torch.long, device=device)
            else:
                bin_idx = torch.bucketize(scores_k, inner_edges, right=False)
                bin_idx = bin_idx.clamp(0, M - 1)

            theta_k = self.hb[k].to(device=device, dtype=dtype)
            calibrated[:, k] = theta_k[bin_idx]

        row_sums = calibrated.sum(dim=1, keepdim=True).clamp_min(self.eps)
        calibrated = calibrated / row_sums

        return calibrated


# Isotonic Binning
class IsotonicRegressionCalibrator:
    def __init__(self):
        self.eps = 1e-12

        self.ir = None  
        self.prior = None
        self.num_classes = None
        self.is_fitted = False

    def fit(self, probs_val: torch.Tensor, y_val: torch.Tensor):
        """
        Args:
            probs_val: Tensor (N, K) probabilités
            y_val: Tensor (N,) labels entiers dans [0, K-1]
        """
        if probs_val.ndim != 2:
            raise ValueError("probs_val doit être 2D (N,K)")
        if y_val.ndim != 1:
            raise ValueError("y_val doit être 1D (N,)")

        N, K = probs_val.shape
        if y_val.shape[0] != N:
            raise ValueError("probs_val et y_val doivent avoir le même N")

        y_np = y_val.detach().cpu().numpy().astype(np.int64)
        if y_np.min() < 0 or y_np.max() >= K:
            raise ValueError("y_val hors [0, K-1]")

        p_np = probs_val.detach().cpu().numpy().astype(np.float64)

        # prior par classe (fallback)
        prior = np.bincount(y_np, minlength=K).astype(np.float64) / max(N, 1)

        self.ir = []
        for k in range(K):
            yk = (y_np == k).astype(np.float64)

            ir_k = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
            ir_k.fit(p_np[:, k], yk)
            self.ir.append(ir_k)

        self.prior = prior
        self.num_classes = K
        self.is_fitted = True
        return self

    def calibrate(self, probs_test: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs_test: Tensor (N, K)
        Returns:
            Tensor (N, K) calibré puis renormalisé
        """
        if not self.is_fitted or self.ir is None or self.prior is None or self.num_classes is None:
            raise RuntimeError("Call fit() before calibrate().")

        if probs_test.ndim != 2:
            raise ValueError("probs_test doit être 2D (N,K)")

        N, K = probs_test.shape
        if K != self.num_classes:
            raise ValueError(f"probs_test a K={K} classes, attendu {self.num_classes}")

        p_np = probs_test.detach().cpu().numpy().astype(np.float64)
        cal = np.empty_like(p_np)

        for k in range(K):
            pk_cal = self.ir[k].transform(p_np[:, k])

            bad = ~np.isfinite(pk_cal)
            if bad.any():
                pk_cal[bad] = self.prior[k]

            pk_cal = np.clip(pk_cal, 0.0, 1.0)
            cal[:, k] = pk_cal

        # renormalisation
        row_sums = cal.sum(axis=1, keepdims=True)
        row_sums = np.clip(row_sums, self.eps, None)
        cal = cal / row_sums

        return torch.tensor(cal, device=probs_test.device, dtype=probs_test.dtype)


# Matrix Scaling
class ModelWithMatrixScaling(nn.Module):
    def __init__(self, model, logits_dim, device):
        super(ModelWithMatrixScaling, self).__init__()
        self.model = model.to(device)
        self.matrix_scale = torch.nn.Linear(logits_dim, logits_dim).to(device)
        self.device = device

        with torch.no_grad():
            self.matrix_scale.weight.copy_(torch.eye(logits_dim, device=device))
            self.matrix_scale.bias.zero_()

    def forward(self, input):
        logits = self.model(input).float()
        return self.matrix_scale(logits)

    def set_matrix(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list = []
        labels_list = []

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                label = label.to(self.device).long()

                logits = self.model(input)
                logits_list.append(logits.float())
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device).long()

        optimizer = optim.LBFGS(self.matrix_scale.parameters(), lr=0.01, max_iter=200)
        
        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.matrix_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.train(training_mode)
    

# Vector Scaling
class ModelWithVectorScaling(nn.Module):
    def __init__(self, model, logits_dim, device):
        super(ModelWithVectorScaling, self).__init__()
        self.model = model.to(device)
        self.device = device

        # Vector scaling: paramètres diagonaux (scale) + biais
        self.scale = torch.nn.Parameter(torch.ones(logits_dim, device=device))
        self.bias = torch.nn.Parameter(torch.zeros(logits_dim, device=device))

    def forward(self, input):
        logits = self.model(input).float()
        return logits * self.scale + self.bias

    def set_vector(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list = []
        labels_list = []

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                label = label.to(self.device).long()

                logits = self.model(input)
                logits_list.append(logits.float())
                labels_list.append(label)

            logits = torch.cat(logits_list, dim=0).to(self.device)
            labels = torch.cat(labels_list, dim=0).to(self.device).long()

        # Optimiser uniquement scale et bias
        optimizer = optim.LBFGS([self.scale, self.bias], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = nll_criterion(logits * self.scale + self.bias, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.train(training_mode)


# Temperature Scaling
## Temperature Scaling with Cross Entropy
class ModelWithTemperature_CrossEntropy(nn.Module):
    def __init__(self, model, device):
        super(ModelWithTemperature_CrossEntropy, self).__init__()
        self.model = model.to(device)
        self.temperature =  nn.Parameter(torch.tensor([1.5], device=device))
        self.device = device

    def forward(self, input):
        logits = self.model(input).float()
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        nll_criterion = nn.CrossEntropyLoss().to(self.device)

        logits_list = []
        labels_list = []

        training_mode = self.training
        self.eval()

        with torch.no_grad():
            for input, label in valid_loader:
                input = input.to(self.device)
                label = label.to(self.device).long()

                logits = self.model(input)
                logits_list.append(logits.float())
                labels_list.append(label)

            logits = torch.cat(logits_list).to(self.device)
            labels = torch.cat(labels_list).to(self.device).long()

        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=200)

        def closure():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        self.train(training_mode)


## Temperature Scaling with Focal Loss
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss on logits.
    Args:
        gamma (float): focusing parameter
        alpha (None | float | Tensor): class weighting.
            - None: no weighting
            - float: same alpha for all classes (rarely useful)
            - Tensor shape (C,): per-class weights
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)         
        probs = log_probs.exp()                           

        targets = targets.long()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1) 
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)          

        if self.alpha is None:
            alpha_t = 1.0
        else:
            if isinstance(self.alpha, (float, int)):
                alpha_t = float(self.alpha)
            else:
                alpha = self.alpha.to(logits.device, dtype=logits.dtype)
                alpha_t = alpha.gather(0, targets)  

        loss = -alpha_t * (1.0 - pt).pow(self.gamma) * log_pt  

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ModelWithTemperature_Focal(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model.to(device)
        self.temperature =  nn.Parameter(torch.tensor([1.5], device=device))
        self.device = device

    def forward(self, input):
        logits = self.model(input).float()
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    @torch.no_grad()
    def _collect_logits_labels(self, valid_loader):
        logits_list, labels_list = [], []
        for x, y in valid_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            logits_list.append(self.model(x).float())
            labels_list.append(y)
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0).long()
        return logits, labels

    def set_temperature(self, valid_loader, gamma: float = 2.0, alpha=None, lr: float = 1e-2, max_iter: int = 200):
        """
        Fit temperature T by minimizing Focal Loss on the validation set.
        """
        # keep BN/Dropout frozen while collecting logits
        was_training = self.training
        self.eval()

        with torch.no_grad():
            logits, labels = self._collect_logits_labels(valid_loader)

        focal_criterion = FocalLoss(gamma=gamma, alpha=alpha, reduction="mean").to(self.device)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = focal_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # restore training mode
        self.train(was_training)
