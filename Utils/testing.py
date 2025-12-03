import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.metrics import Metric
from Utils.Utils import AverageMeter
from Visualization.Visualization import plot_spectrum_comparison
import os

def get_unwrapped_model(model):
    """Get the underlying model from a wrapped model (e.g., DDP, FSDP)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def test_one_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger, accelerator):
    accelerator.wait_for_everyone()
    model.eval()

    loss_meter = AverageMeter()
    loss_meters = {}
    mae_thickness_nm_meter = AverageMeter() 

    unwrapped_model = get_unwrapped_model(model)
    num_outputs = getattr(unwrapped_model, 'structure_layers', 1)

    metrics_calculator = Metric(num_outputs=num_outputs)
    metrics_calculator.to(accelerator.device)
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            output = model(spectrum)
            loss_dict = criterion(output, batch)

            loss_meter.update(loss_dict["loss"])
            
            # Dynamic loss metering
            for k, v in loss_dict.items():
                if k.startswith('loss_') and isinstance(v, torch.Tensor):
                    if k not in loss_meters:
                        loss_meters[k] = AverageMeter()
                    loss_meters[k].update(v)
            
            pred_thickness = output[0]
            target_thickness = batch['layer_thickness']
            
            all_preds.append(pred_thickness)
            all_targets.append(target_thickness)
            
            mae_normalized = torch.mean(torch.abs(pred_thickness - target_thickness))
            mae_nm = mae_normalized * (200 - 20) 
            mae_thickness_nm_meter.update(mae_nm)

            material_preds = output[1]

            if i == 0 and accelerator.is_main_process:
                # Spectrum Reconstruction and Plotting for the first sample in the first batch
                try:
                    dataset = test_dataloader.dataset
                    
                    # Denormalize thickness
                    min_th, max_th = dataset.thickness_range
                    pred_thickness_denorm = pred_thickness * (max_th - min_th) + min_th
                    
                    # Reconstruct spectrum
                    # compute_spectrum now handles both indices and raw refractive indices
                    R_pred, _ = dataset.compute_spectrum(pred_thickness_denorm, material_preds)
                    
                    if R_pred is not None:
                        # Get data for plotting (first sample)
                        wavelength = dataset.wavelength.values.squeeze()
                        if wavelength.ndim > 1: # If broadcasted, take first row
                            wavelength = wavelength[0]
                            
                        target_spec = spectrum[0]
                        pred_spec = R_pred[0]
                        
                        # Determine save directory (infer from logger file handler or use default)
                        save_dir = None
                        for handler in logger_val.handlers:
                            if hasattr(handler, 'baseFilename'):
                                save_dir = os.path.dirname(handler.baseFilename)
                                break
                        if save_dir is None:
                            save_dir = "experiments/default/plots" # Fallback
                        
                        save_dir = os.path.join(save_dir, "plots")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Plot using Matplotlib (now updated in Visualization.py)
                        plot_spectrum_comparison(
                            wavelength=wavelength,
                            target_spectrum=target_spec,
                            pred_spectrum=pred_spec,
                            title=f"Epoch {epoch} - Spectrum Comparison",
                            save_dir=save_dir,
                            save_name=f"spectrum_epoch_{epoch}"
                        )
                        
                        # Optional: Log to TensorBoard if desired
                        # Since we just saved it, we could read it back or just replicate plotting for TB
                        if tb_logger is not None:
                             import matplotlib.pyplot as plt
                             
                             # Re-create figure for TensorBoard to avoid interfering with saved file
                             fig_tb = plt.figure(figsize=(10, 6))
                             plt.plot(wavelength.cpu().numpy(), target_spec.cpu().numpy(), 'k-', label='Target')
                             plt.plot(wavelength.cpu().numpy(), pred_spec.cpu().numpy(), 'r--', label='Prediction')
                             plt.title(f"Epoch {epoch} - Spectrum Comparison")
                             plt.xlabel("Wavelength (nm)")
                             plt.ylabel("Reflectance")
                             plt.legend()
                             plt.grid(True)
                             tb_logger.add_figure('val/spectrum_comparison', fig_tb, epoch)
                             plt.close(fig_tb)

                except Exception as e:
                    logger_val.error(f"Failed to plot spectrum: {e}")
                    import traceback
                    traceback.print_exc()

    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        avg_metrics = metrics_calculator.metric(all_preds, all_targets)
    else:
        avg_metrics = {}

    if accelerator.is_main_process:
        if tb_logger is not None:
            tb_logger.add_scalar('[val]: loss', loss_meter.avg, epoch + 1)
            tb_logger.add_scalar('[val]: mae_thickness_nm', mae_thickness_nm_meter.avg, epoch + 1)
            
            for k, meter in loss_meters.items():
                tb_logger.add_scalar(f'[val]: {k}', meter.avg, epoch + 1)

            for key, value in avg_metrics.items():
                tb_logger.add_scalar(f'[val]: {key}', value, epoch + 1)
        
        loss_str_parts = [f"Loss: {loss_meter.avg:.4f}"]
        for k, meter in loss_meters.items():
            loss_str_parts.append(f"{k.replace('loss_', '')}: {meter.avg:.4f}")
        loss_str_parts.append(f"MAE(nm): {mae_thickness_nm_meter.avg:.2f}")
        
        logger_val.info(f"Test epoch {epoch}: " + " | ".join(loss_str_parts))
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger_val.info(f"Detailed Metrics: {metrics_str}")

    accelerator.wait_for_everyone()
    return loss_meter.avg
