import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import numpy as np
from Utils.metrics import Metric
from Utils.Utils import AverageMeter
from Visualization.MatplotlibVis import plot_sample_spectrum, save_structure_plots, plot_first_sample

def get_unwrapped_model(model):
    """Get the unwrapped model from a potentially wrapped model (e.g., DataParallel, DistributedDataParallel)."""
    if hasattr(model, 'module'):
        return model.module
    return model

def test_one_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger, accelerator):
    accelerator.wait_for_everyone()
    model.eval()

    loss_meter             = AverageMeter()
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            unwrapped_model = get_unwrapped_model(model)
            if unwrapped_model.name== 'SimpleEncoderNextLayer':
                 output = model(spectrum, layer_thickness=batch['layer_thickness'])
            else:
                 output = model(spectrum)

            loss_dict = criterion(output, batch)

            loss_meter.update(loss_dict["loss"])
                            
            if accelerator.is_main_process:
                if i == 0:
                    unwrapped_model = get_unwrapped_model(model)
                    if unwrapped_model.name== 'SimpleEncoderNextLayer':
                        output_for_plot = unwrapped_model.generate(spectrum)
                    else:
                        output_for_plot = output
                    
                    R_calc, T_calc = test_dataloader.dataset.compute_spectrum(layer_thickness=output_for_plot.to(test_dataloader.dataset.device), material_choice=batch['material_choice'])
                    
                    # Add plots to tensorboard
                    if tb_logger is not None:
                        plot_first_sample(tb_logger, test_dataloader.dataset, batch, R_calc, epoch, output_for_plot)
                
        if tb_logger is not None:
            tb_logger.add_scalar('[val]: loss', loss_meter.avg, epoch + 1)

        logger_val.info(
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.4f} | "
        )
            

    accelerator.wait_for_everyone()
    return loss_meter.avg


def evaluate_test_set(net, test_dataloader, experiment_path, accelerator, logger_test):
    """
    Evaluates the model on the test set, saves plots and loss CSVs.
    """
    plots_dir = os.path.join(experiment_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    sample_losses_file = os.path.join(experiment_path, 'sample_losses.csv')
    avg_loss_file = os.path.join(experiment_path, 'average_loss.csv')
    
    net.eval()
    all_losses = []
    
    # For per-sample loss calculation
    mse_criterion = torch.nn.MSELoss(reduction='none')
    

    dataset = test_dataloader.dataset
    device = dataset.device
    thickness_min = dataset.thickness_range[0]
    thickness_max = dataset.thickness_range[1]
    
    if accelerator.is_main_process:
        logger_test.info("Starting testing...")

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            unwrapped_model = get_unwrapped_model(net)
            if unwrapped_model.name== 'SimpleEncoderNextLayer':
                output = unwrapped_model.generate(spectrum)
            else:
                output = net(spectrum)
            
            # Calculate per-sample loss
            target_thickness = batch['layer_thickness']
            loss_per_element = mse_criterion(output, target_thickness)
            loss_per_sample = loss_per_element.mean(dim=1) # (B,)
            
            loss_per_sample_cpu = loss_per_sample.cpu().numpy()
            all_losses.extend(loss_per_sample_cpu)
            
            # Denormalize thickness
            pred_thickness_denorm = output * (thickness_max - thickness_min) + thickness_min
            target_thickness_denorm = target_thickness * (thickness_max - thickness_min) + thickness_min
            
            # Compute spectrum
            R_calc, T_calc = dataset.compute_spectrum(pred_thickness_denorm.to(device), batch['material_choice'].to(device))
            
            batch_size = output.shape[0]
            for b in range(batch_size):
                idx = i * batch_size + b
                
                # Plot
                wavelengths = dataset.wavelength.values.squeeze().cpu().numpy()
                target_R = batch['R'][b].cpu().numpy()
                pred_R = R_calc[b].cpu().numpy()
                
                plot_sample_spectrum(wavelengths, target_R, pred_R, idx, loss_per_sample_cpu[b], plots_dir)

                # Reconstruct structure for visualization
                save_structure_plots(dataset, batch['material_choice'][b], pred_thickness_denorm[b], target_thickness_denorm[b], plots_dir, f'{idx}', f'{idx}')
                
    # Save losses to CSV
    if accelerator.is_main_process:
        with open(sample_losses_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sample Index', 'Loss'])
            for idx, loss in enumerate(all_losses):
                writer.writerow([idx, loss])
                
        avg_loss = np.mean(all_losses)
        with open(avg_loss_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Average Loss'])
            writer.writerow([avg_loss])
            
        logger_test.info(f"Test finished. Average Loss: {avg_loss}")
        logger_test.info(f"Plots saved to {plots_dir}")
        logger_test.info(f"Losses saved to {sample_losses_file} and {avg_loss_file}")
