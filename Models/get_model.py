import torch
from Models.DirectEncoder import DirectEncoder
from Models.SimpleEncoder import SimpleEncoder
from Models.SimpleEncoderNextLayer import SimpleEncoderNextLayer
from Loss.structure_loss import StructureLoss
from Loss.next_token_loss import NextTokenLoss


def get_model(config, args, device):
    # Inject dataset config into model config

    if config.name == 'DirectEncoder':
        net = DirectEncoder(config=config)
        loss = StructureLoss(**args.losses)
        vae = None
    elif config.name == 'SimpleEncoder':
        net = SimpleEncoder(config=config)
        loss = StructureLoss(**args.losses)
        vae = None
    elif config.name == 'SimpleEncoderNextLayer':
        net = SimpleEncoderNextLayer(config=config)
        loss = NextTokenLoss(config['thickness_range'], config.get('thickness_steps', 1))
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