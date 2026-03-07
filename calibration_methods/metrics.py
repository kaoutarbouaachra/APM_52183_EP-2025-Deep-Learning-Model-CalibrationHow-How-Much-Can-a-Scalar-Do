"""
Code for calibration metrics : 
    - ECE
    - MCE
    - Brier Score
    - Classwise ECE 
    - Adaptive ECE
    - Reliability diagrams
"""
import torch


class CalibrationMetrics():
    """
    Args: 
        - bins : list/1D tensor of bin edges (length M+1), e.g. [0.0, 0.1, ..., 1.0]
        - true_labels : a tensor of the ground-truth labels for calculating metrics, it should be a matrix of shape 
            (num_samples, num_classes) with a one hot encoder for the corresponding true class.
    """
    def __init__(self, bins, true_labels, device):
        self.bins = torch.as_tensor(bins, dtype=torch.float32).to(device) 
        self.true_labels = true_labels.to(device)
        self.device = device

    def get_metrics(self, probs):
        return {
            "ece" : self.calculate_ece(probs).item(),
            "mce" : self.calculate_mce(probs).item(),
            "brier_score" : self.calculate_brier_score(probs).item(),
            "cece" : self.calculate_cece(probs).item(),
        }

    def calculate_ece(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
        """
        # predicted class with corresponding confidence
        probs = probs.to(self.device)
        conf, pred = probs.max(dim=1)  

        # true class index 
        true = self.true_labels.argmax(dim=1)  

        correct = (pred == true).float() 
        N = probs.shape[0]

        ece = torch.zeros((), device=self.device)

        M = self.bins.numel() - 1
        for m in range(M):
            lower_bound, upper_bound = self.bins[m], self.bins[m + 1]

            if m == 0:
                in_bin = (conf >= lower_bound) & (conf <= upper_bound)
            else:
                in_bin = (conf > lower_bound) & (conf <= upper_bound)

            bin_count = in_bin.sum()
            if bin_count.item() == 0:
                continue

            acc_bm = correct[in_bin].mean()
            conf_bm = conf[in_bin].mean()

            ece += bin_count.float() * torch.abs(acc_bm - conf_bm)

        ece = ece / float(N)
        return ece
    

    def calculate_mce(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
        """
        # predicted class with corresponding confidence
        probs = probs.to(self.device)
        conf, pred = probs.max(dim=1)  

        # true class index 
        true = self.true_labels.argmax(dim=1)  

        correct = (pred == true).float() 

        mce = torch.zeros((), device=self.device) 

        M = self.bins.numel() - 1
        for m in range(M):
            lower_bound, upper_bound = self.bins[m], self.bins[m + 1]

            if m == 0:
                in_bin = (conf >= lower_bound) & (conf <= upper_bound)
            else:
                in_bin = (conf > lower_bound) & (conf <= upper_bound)

            bin_count = in_bin.sum()
            if bin_count.item() == 0:
                continue

            acc_bm = correct[in_bin].mean()
            conf_bm = conf[in_bin].mean()

            mce = torch.maximum(mce, torch.abs(acc_bm - conf_bm))

        return mce
    

    def calculate_brier_score(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
        """
        num_classes = probs.shape[1]

        if num_classes == 2:
            return self.calculate_binary_bs(probs)
        else:
            return self.calculate_multi_class_bs(probs)


    def calculate_binary_bs(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
            where num_classes = 2
        """
        probs = probs.to(self.device)

        # Probability of positive class 
        f = probs[:, 1]  

        # true class index  
        y = self.true_labels.argmax(dim=1).float()  

        # Brier score
        bs = torch.mean((f - y) ** 2)

        return bs
    

    def calculate_multi_class_bs(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
        """
        probs = probs.to(self.device)
        y = self.true_labels  

        # Brier score
        bs = torch.mean(torch.sum((probs - y) ** 2, dim=1))

        return bs
    

    def calculate_cece(self, probs):
        """
        Args:
            - probs : Tensor of probabilities of each class, a matrix of shape (num_samples, num_classes)
        """
        probs = probs.to(self.device)
        N, K = probs.shape

        cece = torch.zeros((), device=self.device)
        M = self.bins.numel() - 1

        for c in range(K):
            conf_c = probs[:, c]      
            y_c = self.true_labels[:, c].float()     

            ece_c = torch.zeros((), device=self.device)

            for m in range(M):
                lower_bound, upper_bound = self.bins[m], self.bins[m + 1]

                if m == 0:
                    in_bin = (conf_c >= lower_bound) & (conf_c <= upper_bound)
                else:
                    in_bin = (conf_c > lower_bound) & (conf_c <= upper_bound)

                bin_count = in_bin.sum()
                if bin_count.item() == 0:
                    continue

                acc_bm_c = y_c[in_bin].mean() 
                conf_bm_c = conf_c[in_bin].mean()

                ece_c +=  (bin_count.float() / float(N)) * torch.abs(acc_bm_c - conf_bm_c)

            cece += ece_c

        cece = cece / float(K)
        return cece        


