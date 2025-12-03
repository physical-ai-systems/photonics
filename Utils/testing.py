import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.metrics import Metric
from Utils.Utils import AverageMeter
import os


def test_one_epoch(epoch, test_dataloader, model, criterion, logger_val, tb_logger, accelerator):
    accelerator.wait_for_everyone()
    model.eval()

    loss_meter             = AverageMeter()
    
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            spectrum = batch['R'].float()
            
            output = model(spectrum)
            loss_dict = criterion(output, batch)

            loss_meter.update(loss_dict["loss"])
                        
    if accelerator.is_main_process:
        if i == 0:
            R_calc, T_calc = test_dataloader.dataset.compute_spectrum(layer_thickness=output.to(test_dataloader.dataset.device), material_choice=batch['material_choice'])
            
            # Add plots to tensorboard
            if tb_logger is not None:
                # Add histogram of calculated spectra
                tb_logger.add_histogram('[val]: R_calc', R_calc, epoch + 1)
                tb_logger.add_histogram('[val]: T_calc', T_calc, epoch + 1)

                # Add histogram of original input spectra
                if 'R' in batch:
                    tb_logger.add_histogram('[val]: R_orig', batch['R'], epoch + 1)
                if 'T' in batch:
                    tb_logger.add_histogram('[val]: T_orig', batch['T'], epoch + 1)
                
                # Add histogram of output (layer thickness)
                tb_logger.add_histogram('[val]: layer_thickness', output, epoch + 1)
            
            

        
        if tb_logger is not None:
            tb_logger.add_scalar('[val]: loss', loss_meter.avg, epoch + 1)

        logger_val.info(
            f"Test epoch {epoch}: "
            f"Loss: {loss_meter.avg:.4f} | "
        )
        

    accelerator.wait_for_everyone()
    return loss_meter.avg
