import torch
import torch.nn as nn

class StructureLoss(nn.Module):
    def __init__(self, lambda_thickness=1.0, lambda_material=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.material_loss = nn.BCEWithLogitsLoss()
        self.lambda_thickness = lambda_thickness
        self.lambda_material = lambda_material
        # Initialize log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(2))

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

        # Material Loss (BCE with Logits)
        # material_logits: (B, N, C), material_target: (B, N)
        loss_material = self.material_loss(material_logits, material_target.float())

        # Weighting based on uncertainty
        total_loss = self.lambda_thickness * (loss_thickness * torch.exp(-self.log_vars[0]) + self.log_vars[0]) + \
               self.lambda_material * (loss_material * torch.exp(-self.log_vars[1]) + self.log_vars[1])

        return {
            "loss": total_loss,
            "loss_thickness": loss_thickness,
            "loss_material": loss_material
        }

class RefractiveIndexLoss(nn.Module):
    def __init__(self, lambda_thickness=1.0, lambda_refractive=1.0, lambda_quantizer=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_thickness = lambda_thickness
        self.lambda_refractive = lambda_refractive
        self.lambda_quantizer = lambda_quantizer
        # Initialize log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(2))

    def forward(self, output, target):
        """
        output: tuple (thickness_pred, refractive_indices_pred, vq_loss)
        target: dict {'layer_thickness': ..., 'refractive_indices': ...}
        """
        vq_loss = 0
        if len(output) == 3:
             thickness_pred, refractive_indices_pred, vq_loss = output
        else:
             thickness_pred, refractive_indices_pred = output
        
        thickness_target = target['layer_thickness']
        refractive_indices_target = target['refractive_indices']

        # Thickness Loss (MSE)
        loss_thickness = self.mse(thickness_pred, thickness_target)

        # Refractive Index Loss (MSE on n and k)
        loss_refractive = self.mse(refractive_indices_pred, refractive_indices_target)

        # Weighting based on uncertainty
        total_loss = self.lambda_thickness * (loss_thickness * torch.exp(-self.log_vars[0]) + self.log_vars[0]) + \
               self.lambda_refractive * (loss_refractive * torch.exp(-self.log_vars[1]) + self.log_vars[1]) + \
               self.lambda_quantizer * vq_loss

        return {
            "loss": total_loss,
            "loss_thickness": loss_thickness,
            "loss_refractive": loss_refractive,
            "loss_quantizer": vq_loss
        }