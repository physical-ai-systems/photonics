import torch.optim as optim


def configure_optimizers(net, args):
    """Configure and return the main optimizer only."""

    parameters = [
        p for n, p in net.named_parameters() if p.requires_grad
    ]
    optimizer = optim.AdamW(
        parameters,
        lr=float(args.learning_rate),
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    )
    return optimizer