import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torchmetrics.functional import (
    cosine_similarity,
    explained_variance,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    pearson_corrcoef,
    r2_score,
    spearman_corrcoef,
    symmetric_mean_absolute_percentage_error,
    tweedie_deviance_score,
    weighted_mean_absolute_percentage_error
)


class Metric(nn.Module):
    def __init__(self, num_outputs=1):
        super(Metric, self).__init__()
        
    def metric(self, output, target):

        with torch.no_grad():
            cs = cosine_similarity(output, target, reduction='mean').item()
            ev = explained_variance(output, target).item()
            mae = mean_absolute_error(output, target).item()
            mape = mean_absolute_percentage_error(output, target).item()
            mse = mean_squared_error(output, target, squared=True).item()
            msle = mean_squared_log_error(output, target).item()
            
            pcc_val = pearson_corrcoef(output, target)
            pcc = pcc_val.mean().item() if pcc_val.numel() > 1 else pcc_val.item()
            
            r2s = r2_score(output, target).item()
            
            scc_val = spearman_corrcoef(output, target)
            scc = scc_val.mean().item() if scc_val.numel() > 1 else scc_val.item()

            smape = symmetric_mean_absolute_percentage_error(output, target).item()
            tds = tweedie_deviance_score(output, target).item()
            wmape = weighted_mean_absolute_percentage_error(output, target).item()

            metrics = { 
                'cs': cs,
                'ev': ev,
                'mae': mae,
                'mape': mape,
                'mse': mse,
                'msle': msle,
                'pcc': pcc,
                'r2s': r2s,
                'scc': scc,
                'smape': smape,
                'tds': tds,
                'wmape': wmape
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
    metrics_col = ['model_name', 'quality', 'dataset_name', 'cs', 'ev', 'mae', 'mape', 'mse', 'msle', 'pcc', 'r2s', 'scc', 'smape', 'tds', 'wmape']
    if overwrite or not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            f.write(','.join(metrics_col) + '\n')
    with open(csv_path, 'a') as f:
        row = [str(metrics.get(col, '')) for col in metrics_col]
        f.write(','.join(row) + '\n')