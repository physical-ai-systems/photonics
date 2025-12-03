import torch
import torch.nn as nn

class StructureLoss(nn.Module):
    def __init__(self, lambda_thickness=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_thickness = lambda_thickness

    def forward(self, output, target):
        """
        output: thickness_pred
        target: dict {'layer_thickness': ..., 'material_choice': ...}
        """
        thickness_pred = output
        
        thickness_target = target['layer_thickness']

        # Thickness Loss (MSE)
        loss_thickness = self.mse(thickness_pred, thickness_target)

        # Weighting based on uncertainty
        total_loss = loss_thickness 

        return {
            "loss": total_loss,
            "loss_thickness": loss_thickness,
        }
