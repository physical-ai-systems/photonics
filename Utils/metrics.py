import os
import torch
import torch.nn as nn
import numpy as np
import scipy.interpolate
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
        
    


def bjontegaard_metric_psnr(R1, PSNR1, R2, PSNR2, piecewise=0):
    '''
    This function is taken from:https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    '''

    lR1 = np.log(R1)
    lR2 = np.log(R2)

    PSNR1 = np.array(PSNR1)
    PSNR2 = np.array(PSNR2)

    p1 = np.polyfit(lR1, PSNR1, 3)
    p2 = np.polyfit(lR2, PSNR2, 3)

    # integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        # See https://chromium.googlesource.com/webm/contributor-guide/+/master/scripts/visual_metrics.py
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(lR1), PSNR1[np.argsort(lR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(lR2), PSNR2[np.argsort(lR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapezoid(v1, dx=interval)
        int2 = np.trapezoid(v2, dx=interval)

    # find avg diff
    avg_diff = (int2-int1)/(max_int-min_int)

    return avg_diff


def bjontegaard_metric_rate(R1, PSNR1, R2, PSNR2, piecewise=0):
    '''
    This function is taken from:https://github.com/Anserw/Bjontegaard_metric/blob/master/bjontegaard_metric.py
    '''
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # rate method
    p1 = np.polyfit(PSNR1, lR1, 3)
    p2 = np.polyfit(PSNR2, lR2, 3)

    # integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # find integral
    if piecewise == 0:
        p_int1 = np.polyint(p1)
        p_int2 = np.polyint(p2)

        int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
        int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)
    else:
        lin = np.linspace(min_int, max_int, num=100, retstep=True)
        interval = lin[1]
        samples = lin[0]
        v1 = scipy.interpolate.pchip_interpolate(np.sort(PSNR1), lR1[np.argsort(PSNR1)], samples)
        v2 = scipy.interpolate.pchip_interpolate(np.sort(PSNR2), lR2[np.argsort(PSNR2)], samples)
        # Calculate the integral using the trapezoid method on the samples.
        int1 = np.trapz(v1, dx=interval)
        int2 = np.trapz(v2, dx=interval)

    # find avg diff
    avg_exp_diff = (int2-int1)/(max_int-min_int)
    avg_diff = (np.exp(avg_exp_diff)-1)*100
    return avg_diff

