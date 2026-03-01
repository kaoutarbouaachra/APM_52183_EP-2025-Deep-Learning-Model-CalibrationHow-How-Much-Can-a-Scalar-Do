import numpy as np
import torch
class TemperatureScaledModel(torch.nn.Module):

    def __init__(self, model, temp_init=1, device="cpu") -> None:
        super().__init__()
        self.model = model
        self.device = device
        self.temperature = torch.tensor(temp_init * np.ones(1), device=device, requires_grad=True)
    
    def forward(self, x):
        return self.model(x) / self.temperature
    
    def fit(self, cal_data_loader):
        # First pre-compute all logits.
        logits, labels = [], []
        with torch.no_grad():
            for data, target in cal_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                logits.append(self.model(data))
                labels.append(target)
            logits = torch.cat(logits).to(self.device)
            labels = torch.cat(labels).to(self.device)
        
        # Fit temperature.
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)

        def temp_eval():
            optimizer.zero_grad()
            loss = criterion(logits / self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(temp_eval)

