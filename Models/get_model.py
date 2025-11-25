import torch
# from Models import * # This wildcard import is causing issues if Models.py is empty or __init__.py is missing
from Models.DirectEncoder import DirectEncoder
from Loss.rd_loss import RateDistortionLoss, RateDistortionLoss1dToken
from Loss.structure_loss import StructureLoss

# Placeholder imports for other models if they exist in other files, 
# otherwise we should comment them out or import them correctly.
# Assuming they might be in base_model.py or similar based on file list, 
# but since the user only cares about DirectEncoder now, we can leave them as is 
# but we need to make sure the code runs.
# If Models.py is empty, 'from Models import *' does nothing useful and might hide errors.

def get_model(config, args, device):
    if config.Model == 'DirectEncoder':
        net = DirectEncoder(config=config.model_params)
        loss = StructureLoss(**config.loss_params)
        vae = None
    # Commenting out other models for now to avoid NameErrors if they are not defined
    # elif config.Model == 'opod_tic':
    #     net=Model(config=config)
    #     loss = RateDistortionLoss1dToken(**args.losses)
    #     vae = None
    # elif config.Model == 'LoC_LIC':
    #     net=LoC_LIC(config=config)
    #     loss = RateDistortionLoss(lmbda=args.lmbda)
    #     vae = None
    # elif config.Model == 'LoC_LIC_Lite':
    #     net = LoC_LIC_Lite(config=config)
    #     loss = RateDistortionLoss(lmbda=args.lmbda)
    #     vae = None
    # elif config.Model == 'LoC_LIC_residually':
    #     net = LoC_LIC_residually(config=config)
    #     loss = RateDistortionLoss(lmbda=args.lmbda)
    #     vae = None
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