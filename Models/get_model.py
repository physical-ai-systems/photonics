import torch
from Models.DirectEncoder import DirectEncoder
from Loss.structure_loss import StructureLoss


def get_model(config, args, device):
    if config.Model == 'DirectEncoder':
        net = DirectEncoder(config=config.model_params)
        loss = StructureLoss(**config.loss_params)
        vae = None
    else:
        raise ValueError(f"Model {config.Model} not found or not supported in this cleanup.")
    return net, vae, loss

def get_schedulers(optimizer, aux_optimizer, args):
    """
    Returns learning rate schedulers for the main and auxiliary optimizers.
    """
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )
    
    lr_scheduler_aux = None
    if aux_optimizer is not None:
        lr_scheduler_aux = torch.optim.lr_scheduler.ReduceLROnPlateau(
            aux_optimizer, mode='min', factor=0.5, patience=10
        )
        
    return lr_scheduler, lr_scheduler_aux