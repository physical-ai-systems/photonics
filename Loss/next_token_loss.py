import torch
import torch.nn as nn

class NextTokenLoss(nn.Module):
    def __init__(self, thickness_range, thickness_steps):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.thickness_range = thickness_range
        self.thickness_steps = thickness_steps
        self.vocab_size = int((thickness_range[1] - thickness_range[0]) / thickness_steps) + 1

    def forward(self, output, target):
        """
        output: logits (B, L+1, Vocab)
        target: dict {'layer_thickness': (B, L) normalized floats}
        """
        logits = output
        thickness_norm = target['layer_thickness']
        
        # Convert target to indices
        # t_norm in [0, 1]
        # idx = round(t_norm * (vocab_size - 1))
        thickness_indices = torch.round(thickness_norm * (self.vocab_size - 1)).long()
        
        # Align predictions and targets
        # Logits: [Pred_T0, Pred_T1, ..., Pred_TL] (from inputs S, T0, ..., TL-1)
        # Targets: [T0, T1, ..., TL-1]
        # We want to match Pred_T0 with T0, ..., Pred_TL-1 with TL-1.
        # So we take the first L logits.
        
        L = thickness_indices.shape[1]
        preds = logits[:, :L, :] # (B, L, Vocab)
        targets = thickness_indices # (B, L)
        
        loss = self.criterion(preds.reshape(-1, self.vocab_size), targets.reshape(-1))
        
        return {
            "loss": loss,
            "loss_thickness": loss
        }
