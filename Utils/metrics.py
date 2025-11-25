import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure


class ImageMetric(nn.Module):
    def __init__(self):
        super(ImageMetric, self).__init__()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = StructuralSimilarityIndexMeasure()
        self.msssim = MultiScaleStructuralSimilarityIndexMeasure()
        
    def metric_image(self, output, target, size_of_image=None): 
        with torch.no_grad():
            psnr = self.psnr(output, target).item()
            ssim = self.ssim(output, target).item()
            msssim = self.msssim(output, target).item()
                
            metrics = { 
                'psnr': psnr,
                'ssim': ssim, 
                'msssim': msssim,
                    }
            return metrics
    
    
    def append_to_metrics_dict(self, new_metrics: Dict, metrics_dict: Dict=None) -> Dict:
        if metrics_dict is None:
            metrics_dict = {}
        for key, value in new_metrics.items():
            if key in metrics_dict:
                metrics_dict[key].append(value)
            else:
                metrics_dict[key] = [value]
        return metrics_dict
    
    def get_avg_metrics(self, metrics_dict: Dict) -> Dict:
        avg_metrics = {}
        for key, values in metrics_dict.items():
            avg_metrics[key] = np.mean(values)
        return avg_metrics

def write_metrics_to_csv(metrics: Dict, csv_path: str, overwrite: bool=False):
    metrics_col = ['model_name', 'quality', 'dataset_name', 'psnr', 'ssim', 'msssim',]
    if overwrite or not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(','.join(metrics_col) + '\n')
    with open(csv_path, 'a') as f:
        # if the metric is not computed, write an empty string
        row = [str(metrics.get(col, '')) for col in metrics_col]
        f.write(','.join(row) + '\n')
        
