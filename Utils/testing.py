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

    loss_meter             = AverageMeter()
    loss_thickness_meter   = AverageMeter()
    loss_material_meter    = AverageMeter()
    acc_material_meter     = AverageMeter()
    mae_thickness_nm_meter = AverageMeter() 

    unwrapped_model = get_unwrapped_model(model)
    num_outputs = getattr(unwrapped_model, 'output_layers', 1)

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
            loss_thickness_meter.update(loss_dict["loss_thickness"])
            loss_material_meter.update(loss_dict["loss_material"])
            
            pred_thickness = output[0]
            target_thickness = batch['layer_thickness']
            
            all_preds.append(pred_thickness)
            all_targets.append(target_thickness)
            
            mae_normalized = torch.mean(torch.abs(pred_thickness - target_thickness))
            mae_nm = mae_normalized * (200 - 20) 
            mae_thickness_nm_meter.update(mae_nm)

            material_logits = output[1]
            material_preds = material_logits.argmax(dim=-1)
            material_targets = batch['material_choice']
            acc = (material_preds == material_targets).float().mean()
            acc_material_meter.update(acc)

            if i == 0 and accelerator.is_main_process:
                # Spectrum Reconstruction and Plotting for the first sample in the first batch
                try:
                    dataset = test_dataloader.dataset
                    
                    # Denormalize thickness
                    min_th, max_th = dataset.thickness_range
                    pred_thickness_denorm = pred_thickness * (max_th - min_th) + min_th
                    
                    # Reconstruct spectrum
                    R_pred, _ = dataset.compute_spectrum(pred_thickness_denorm, material_preds)
                    
                    # Get data for plotting (first sample)
                    wavelength = dataset.wavelength.values.squeeze()
                    if wavelength.ndim > 1: # If broadcasted, take first row
                        wavelength = wavelength[0]
                        
                    target_spec = spectrum[0]
                    pred_spec = R_pred[0]
                    
                    # Determine save directory (infer from logger file handler or use default)
                    # Assuming logger_val has a file handler, getting its directory
                    save_dir = None
                    for handler in logger_val.handlers:
                        if hasattr(handler, 'baseFilename'):
                            save_dir = os.path.dirname(handler.baseFilename)
                            break
                    if save_dir is None:
                        save_dir = "experiments/default/plots" # Fallback
                    
                    save_dir = os.path.join(save_dir, "plots")
                    os.makedirs(save_dir, exist_ok=True)
                    
                    fig = plot_spectrum_comparison(
                        wavelength=wavelength,
                        target_spectrum=target_spec,
                        pred_spectrum=pred_spec,
                        title=f"Epoch {epoch} - Spectrum Comparison",
                        save_dir=save_dir,
                        save_name=f"spectrum_epoch_{epoch}"
                    )
                    
                    if tb_logger is not None:
                         # Convert plotly fig to image for tensorboard if supported, 
                         # or just log it if tb_logger supports figures.
                         # SummaryWriter.add_figure expects matplotlib figure.
                         # Plotly to image conversion might require kaleido/orca.
                         # For safety/simplicity, we might skip direct TB figure logging if dependencies are tricky
                         # but the user asked for "logged".
                         # Alternative: Log the saved image.
                         import matplotlib.pyplot as plt
                         import io
                         from PIL import Image
                         
                         # Create a matplotlib version for TensorBoard to ensure compatibility
                         plt.figure(figsize=(10, 6))
                         plt.plot(wavelength.cpu().numpy(), target_spec.cpu().numpy(), 'k-', label='Target')
                         plt.plot(wavelength.cpu().numpy(), pred_spec.cpu().numpy(), 'r--', label='Prediction')
                         plt.title(f"Epoch {epoch} - Spectrum Comparison")
                         plt.xlabel("Wavelength (nm)")
                         plt.ylabel("Reflectance")
                         plt.legend()
                         plt.grid(True)
                         tb_logger.add_figure('val/spectrum_comparison', plt.gcf(), epoch)
                         plt.close()

                except Exception as e:
                    logger_val.error(f"Failed to plot spectrum: {e}")
                    import traceback
                    traceback.print_exc()

    # Calculate metrics once at the end of the epoch
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_metrics = metrics_calculator.metric(all_preds, all_targets)

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
