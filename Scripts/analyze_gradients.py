
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import yaml
sys.path.append(os.getcwd())

from Models.SimpleEncoderNextLayer import SimpleEncoderNextLayer

def analyze_model():
    yaml_config_path = 'configs/models/SimpleEncoderNextLayer.yaml'
    with open(yaml_config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = {
        'spectrum_len': 301,
        'thickness_range': [0, 50],
        'structure_layers': 32

    }
    config.update(yaml_config)
    
    print(f"Initializing model with config: {config}")
    model = SimpleEncoderNextLayer(config)
    
    batch_size = 4
    spectrum = torch.randn(batch_size, config['spectrum_len'])
    vocab_size = model.thickness_vocab_size
    layer_thickness = torch.randint(0, vocab_size, (batch_size, config['structure_layers'])).float() / (vocab_size - 1)
    
    print("Running forward pass...")
    model.train()
    logits = model(spectrum, layer_thickness)
    
    print("Running backward pass...")
    target = torch.randint(0, vocab_size, logits.shape[:-1]).to(logits.device)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits.view(-1, vocab_size), target.view(-1))
    loss.backward()
    print("Collecting statistics...")
    stats = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Weights
            w_mean = param.data.mean().item()
            w_var = param.data.var().item()
            w_min = param.data.min().item()
            w_max = param.data.max().item()
            
            # Gradients
            if param.grad is not None:
                g_mean = param.grad.mean().item()
                g_var = param.grad.var().item()
                g_min = param.grad.min().item()
                g_max = param.grad.max().item()
                g_norm = param.grad.norm().item()
                
                # Check for issues
                issue = []
                if np.isnan(g_mean) or np.isinf(g_mean):
                    issue.append("NaN/Inf Gradient")
                elif g_var < 1e-8:
                    issue.append("Vanishing Gradient (Var < 1e-8)")
                elif g_var > 10.0:
                    issue.append("Exploding Gradient (Var > 10)")
                
                stats.append({
                    'layer': name,
                    'weight_mean': w_mean,
                    'weight_var': w_var,
                    'grad_mean': g_mean,
                    'grad_var': g_var,
                    'grad_min': g_min,
                    'grad_max': g_max,
                    'grad_norm': g_norm,
                    'issues': ", ".join(issue)
                })
            else:
                stats.append({
                    'layer': name,
                    'weight_mean': w_mean,
                    'weight_var': w_var,
                    'grad_mean': None,
                    'grad_var': None,
                    'issues': "No Gradient"
                })


    output_path = 'Docs/Gradient_Analysis_SimpleEncoderNextLayer.md'
    with open(output_path, 'w') as f:
        f.write("# Gradient & Weight Analysis: SimpleEncoderNextLayer\n\n")
        f.write("## Experimental Setup\n")
        f.write("### Configuration\n")
        f.write("| Parameter | Value |\n| :--- | :--- |\n")
        for k, v in config.items():
            f.write(f"| `{k}` | `{v}` |\n")
        f.write("\n")
        f.write(f"- **Batch Size**: {batch_size}\n")
        f.write(f"- **Initialization**: Random (trunc_normal_ std=0.02)\n\n")
        
        f.write("## Potential Issues Detected\n")
        issues_found = [s for s in stats if s['issues']]
        if issues_found:
            f.write("| Layer | Issue |\\n|---|---|\\n")
            for s in issues_found:
                f.write(f"| `{s['layer']}` | {s['issues']} |\n")
        else:
            f.write("No obvious vanishing (Var < 1e-8) or exploding (Var > 10) gradients detected in this pass.\n")
            
        f.write("\n## Detailed Layer Statistics\n\n")
        f.write("| Layer Name | Weight Mean | Weight Var | Grad Mean | Grad Var | Grad Norm | Notes |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        
        for s in stats:
            g_mean = f"{s['grad_mean']:.2e}" if s['grad_mean'] is not None else "N/A"
            g_var = f"{s['grad_var']:.2e}" if s['grad_var'] is not None else "N/A"
            g_norm = f"{s['grad_norm']:.2e}" if s.get('grad_norm') is not None else "N/A"
            
            f.write(f"| `{s['layer']}` | {s['weight_mean']:.2e} | {s['weight_var']:.2e} | {g_mean} | {g_var} | {g_norm} | {s['issues']} |\n")
            
    print(f"Analysis complete. Report written to {output_path}")

if __name__ == "__main__":
    analyze_model()
