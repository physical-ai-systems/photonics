import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.metrics import Metric
from Utils.Utils import AverageMeter

def get_unwrapped_model(model):
    """Get the underlying model from a wrapped model (e.g., DDP, FSDP)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def test_one_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger, accelerator):
    accelerator.wait_for_everyone()
    model.eval()

    loss_meter             = AverageMeter()
    loss_thickness_meter   = AverageMeter()
    loss_material_meter    = AverageMeter()
    acc_material_meter     = AverageMeter()
    mae_thickness_nm_meter = AverageMeter() 

    unwrapped_model = get_unwrapped_model(model)
    num_outputs = getattr(unwrapped_model, 'output_layers', 1)

    metrics_calculator = Metric(num_outputs=num_outputs)
    metrics_calculator.to(accelerator.device)
    accumulated_metrics = {}

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            output = model(spectrum)
            loss_dict = criterion(output, batch)

            loss_meter.update(loss_dict["loss"])
            loss_thickness_meter.update(loss_dict["loss_thickness"])
            loss_material_meter.update(loss_dict["loss_material"])
            
            pred_thickness = output[0]
            target_thickness = batch['layer_thickness']
            
            metrics = metrics_calculator.metric(pred_thickness, target_thickness)
            accumulated_metrics = metrics_calculator.append_to_metrics_dict(metrics, accumulated_metrics)
            
            mae_normalized = torch.mean(torch.abs(pred_thickness - target_thickness))
            mae_nm = mae_normalized * (200 - 20) 
            mae_thickness_nm_meter.update(mae_nm)

            material_logits = output[1]
            material_preds = material_logits.argmax(dim=-1)
            material_targets = batch['material_choice']
            acc = (material_preds == material_targets).float().mean()
            acc_material_meter.update(acc)

    avg_metrics = metrics_calculator.get_avg_metrics(accumulated_metrics)

    if accelerator.is_main_process:
        if tb_logger is not None:
            tb_logger.add_scalar('[val]: loss', loss_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: loss_thickness', loss_thickness_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: loss_material', loss_material_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: mae_thickness_nm', mae_thickness_nm_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: acc_material', acc_material_meter.avg, epoch + 1)

            for key, value in avg_metrics.items():
                tb_logger.add_scalar(f'[val]: {key}', value, epoch + 1)

        logger_val.info(
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Thick: {loss_thickness_meter.avg:.4f} | "
            f"Mat: {loss_material_meter.avg:.4f} | "
            f"MAE(nm): {mae_thickness_nm_meter.avg:.2f} | "
            f"Acc: {acc_material_meter.avg:.4f}"
        )
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger_val.info(f"Detailed Metrics: {metrics_str}")

    accelerator.wait_for_everyone()
    return loss_meter.avg
