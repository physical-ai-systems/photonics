import os
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
from torchmetrics.regression import CosineSimilarity, ExplainedVariance, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError, MeanSquaredLogError, PearsonCorrCoef, R2Score, SpearmanCorrCoef, SymmetricMeanAbsolutePercentageError, TweedieDevianceScore, WeightedMeanAbsolutePercentageError


class Metric(nn.Module):
    def __init__(self, num_outputs=1):
        super(Metric, self).__init__()
        self.cs = CosineSimilarity(reduction='mean')
        self.ev = ExplainedVariance()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.mse = MeanSquaredError(squared=True)
        self.msle = MeanSquaredLogError()
        self.pcc = PearsonCorrCoef(num_outputs=num_outputs)
        self.r2s = R2Score()
        self.scc = SpearmanCorrCoef(num_outputs=num_outputs)
        self.smape = SymmetricMeanAbsolutePercentageError()
        self.tds = TweedieDevianceScore()
        self.wmape = WeightedMeanAbsolutePercentageError()
        
    def metric(self, output, target): 
    #     for module in self.children():
    #         if hasattr(module, 'reset'):
    #             module.reset()

        with torch.no_grad():
            cs = self.cs(output, target).item()
            ev = self.ev(output, target).item()
            mae = self.mae(output, target).item()
            mape = self.mape(output, target).item()
            mse = self.mse(output, target).item()
            msle = self.msle(output, target).item()
            
            pcc_val = self.pcc(output, target)
            pcc = pcc_val.mean().item() if pcc_val.numel() > 1 else pcc_val.item()
            
            r2s = self.r2s(output, target).item()
            
            scc_val = self.scc(output, target)
            scc = scc_val.mean().item() if scc_val.numel() > 1 else scc_val.item()

            smape = self.smape(output, target).item()
            tds = self.tds(output, target).item()
            wmape = self.wmape(output, target).item()

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
        
