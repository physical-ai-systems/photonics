import torch
import torch.nn as nn

class StructureLoss(nn.Module):
    def __init__(self, lambda_thickness=1.0, lambda_material=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_thickness = lambda_thickness
        self.lambda_material = lambda_material

    def forward(self, output, target):
        """
        output: tuple (thickness_pred, material_logits)
        target: dict {'layer_thickness': ..., 'material_choice': ...}
        """
        thickness_pred, material_logits = output
        
        thickness_target = target['layer_thickness']
        material_target = target['material_choice']

        # Thickness Loss (MSE)
        loss_thickness = self.mse(thickness_pred, thickness_target)

        # Material Loss (Cross Entropy)
        # material_logits: (B, N, C), material_target: (B, N)
        # Permute logits to (B, C, N) for CrossEntropyLoss
        loss_material = self.ce(material_logits.permute(0, 2, 1), material_target)

        total_loss = (self.lambda_thickness * loss_thickness) + (self.lambda_material * loss_material)

        return {
            "loss": total_loss,
            "loss_thickness": loss_thickness,
            "loss_material": loss_material
        }
