import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.metrics import ImageMetric
from Utils.Utils import AverageMeter
import random


def get_unwrapped_model(model):
    """Get the underlying model from a wrapped model (e.g., DDP, FSDP)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def test_one_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger, accelerator):
    accelerator.wait_for_everyone()
    model.eval()

    loss_meter           = AverageMeter()
    loss_thickness_meter = AverageMeter()
    loss_material_meter  = AverageMeter()
    acc_material_meter   = AverageMeter()
    
    # Metrics for real-world interpretation
    mae_thickness_nm_meter = AverageMeter() # Mean Absolute Error in nanometers

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            output = model(spectrum)
            loss_dict = criterion(output, batch)

            loss_meter.update(loss_dict["loss"])
            loss_thickness_meter.update(loss_dict["loss_thickness"])
            loss_material_meter.update(loss_dict["loss_material"])
            
            # Calculate MAE in nm
            # Output thickness is normalized [0, 1]
            # Target thickness is normalized [0, 1]
            # We need to denormalize to get nm error
            # Assuming range is [20, 200] as per Dataset default
            # Ideally we should get this from dataset, but for now hardcode or pass as arg
            # Let's assume standard range for now: 180nm span
            
            pred_thickness = output[0]
            target_thickness = batch['layer_thickness']
            
            mae_normalized = torch.mean(torch.abs(pred_thickness - target_thickness))
            mae_nm = mae_normalized * (200 - 20) # Denormalize
            
            mae_thickness_nm_meter.update(mae_nm)

            # Calculate Material Accuracy
            material_logits = output[1] # (B, Layers, 2)
            material_preds = material_logits.argmax(dim=-1) # (B, Layers)
            material_targets = batch['material_choice'] # (B, Layers)
            acc = (material_preds == material_targets).float().mean()
            acc_material_meter.update(acc)

    # Only log on main process
    if accelerator.is_main_process:
        if tb_logger is not None:
            tb_logger.add_scalar('[val]: loss', loss_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: loss_thickness', loss_thickness_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: loss_material', loss_material_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: mae_thickness_nm', mae_thickness_nm_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: acc_material', acc_material_meter.avg, epoch + 1)

        logger_val.info(
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Thick: {loss_thickness_meter.avg:.4f} | "
            f"Mat: {loss_material_meter.avg:.4f} | "
            f"MAE(nm): {mae_thickness_nm_meter.avg:.2f} | "
            f"Acc: {acc_material_meter.avg:.4f}"
        )   

    accelerator.wait_for_everyone()
    return loss_meter.avg
