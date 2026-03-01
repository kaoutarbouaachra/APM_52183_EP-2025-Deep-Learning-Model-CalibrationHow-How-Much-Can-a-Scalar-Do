import numpy as np
import torch
class GeneralMixupLoss(torch.nn.Module):
    """
    Custom loss module implementing a generalized Mixup strategy.

    Instead of training on pure (data, target) pairs,
    the model is trained on convex combinations of multiple samples.

    The mixing weights are drawn from a Dirichlet distribution.
    """

    def __init__(self, criterion, alpha=1, mix_size=2):
        """
        Parameters
        ----------
        criterion : torch loss function
            The base loss function (e.g., CrossEntropyLoss).

        alpha : float
            Concentration parameter of the Dirichlet distribution.
            Controls how uniform or peaked the mixing weights are.

        mix_size : int
            Number of samples to mix together.
            mix_size = 2 corresponds to standard Mixup.
        """
        super().__init__()

        self.criterion = criterion
        self.alpha = alpha
        self.mix_size = mix_size

    def forward(self, model, data, target):
        """
        Forward pass for Mixup training.

        Steps:
        1. Sample mixing weights from a Dirichlet distribution.
        2. Create a weighted combination of shuffled inputs.
        3. Compute the weighted loss over mixed targets.
        """

        # Sample mixing weights (sum to 1)
        weights = np.random.dirichlet([self.alpha] * self.mix_size)

        # Initialize mixed data with first component
        mix_data = weights[0] * data
        all_targets = [target]

        # Add additional shuffled components
        for i in range(1, self.mix_size):

            # Random permutation of batch indices
            shuffle = torch.randperm(len(target))

            # Add weighted shuffled data
            mix_data += weights[i] * data[shuffle]

            # Store corresponding shuffled targets
            all_targets.append(target[shuffle].clone())

        # Forward pass on mixed inputs
        out = model(mix_data)

        # Compute weighted sum of losses
        total_loss = 0
        for i, weight in enumerate(weights):
            total_loss += weight * self.criterion(out, all_targets[i]).mean()

        return total_loss
    

