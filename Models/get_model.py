import torch
from Models.DirectEncoder import DirectEncoder
from Models.SimpleEncoder import SimpleEncoder
from Models.RefractiveEncoder import RefractiveEncoder
from Loss.structure_loss import StructureLoss, RefractiveIndexLoss


def get_model(config, args, device):
    if config.name == 'DirectEncoder':
        net = DirectEncoder(config=config)
        loss = StructureLoss(**args.losses)
        vae = None
    elif config.name == 'SimpleEncoder':
        net = SimpleEncoder(config=config)
        loss = StructureLoss(**args.losses)
        vae = None
    elif config.name == 'RefractiveEncoder':
        net = RefractiveEncoder(config=config)
        # Map arguments for RefractiveIndexLoss
        loss_args = {}
        if 'lambda_thickness' in args.losses:
            loss_args['lambda_thickness'] = args.losses['lambda_thickness']
        if 'lambda_refractive' in args.losses:
            loss_args['lambda_refractive'] = args.losses['lambda_refractive']
        elif 'lambda_material' in args.losses: # Fallback/Mapping
            loss_args['lambda_refractive'] = args.losses['lambda_material']
        
        if 'lambda_quantizer' in args.losses:
            loss_args['lambda_quantizer'] = args.losses['lambda_quantizer']
            
        loss = RefractiveIndexLoss(**loss_args)
        vae = None
    else:
        raise ValueError(f"Model {config.name} not found or not supported in this cleanup.")
    return net, vae, loss

def get_schedulers(optimizer, args):
    """
    Returns learning rate schedulers for the main and auxiliary optimizers.
    """
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    ) 
    return lr_scheduler