import os
import sys
import torch
import numpy as np
from scipy.interpolate import interp1d
from accelerate import Accelerator

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.get_model import get_model
from Utils.config import model_config
from Utils.args import test_options
from Dataset.TMM_Fast import PhotonicDatasetTMMFast

class InferenceModel:
    def __init__(self, experiment_name='test_experiment', config_name='simple_encoder'):
        self.repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load args
        self.args = test_options(args=[])

        # Load config
        config_path = os.path.join(self.repo_path, 'configs', 'models', config_name + '.yaml')
        self.config = model_config(config_path, self.args)
        
        # Initialize model
        self.net, _, _ = get_model(self.config, self.args, self.device)
        
        # Initialize Accelerator for loading
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.net = self.accelerator.prepare(self.net)
        
        # Load checkpoint
        # Check if experiment_name is a direct path to a checkpoint
        direct_checkpoint_path = os.path.join(self.repo_path, 'experiments', experiment_name)
        # default_checkpoint_path = os.path.join(self.repo_path, 'experiments', experiment_name, 'checkpoints', 'checkpoint_best_loss')
        default_checkpoint_path = os.path.join(self.repo_path, 'experiments', 'checkpoint_best_loss')
        
        if os.path.exists(os.path.join(direct_checkpoint_path, 'pytorch_model.bin')):
            checkpoint_path = direct_checkpoint_path
        else:
            checkpoint_path = default_checkpoint_path

        if os.path.exists(checkpoint_path):
            try:
                self.accelerator.load_state(checkpoint_path)
                print(f"Loaded checkpoint from {checkpoint_path}")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}, using random weights")
            
        self.net.eval()
        
        # Define target wavelengths (should match training)
        # From log: ranges=[400, 700], steps=1
        self.target_wavelengths = np.arange(400, 701, 1)
        
        # Material mapping (from TMM_Fast.py)
        self.materials = ["SiO2", "Air"]
        self.refractive_indices = {"SiO2": 1.4618, "Air": 1.0}
        self.colors = {"SiO2": "#4839B7", "Air": "#E0E0E0"}

        # Initialize TMM for spectrum calculation
        self.tmm = PhotonicDatasetTMMFast(
            structure_layers=self.config.structure_layers,
            ranges=(400, 700),
            steps=1,
            device=('cuda' if torch.cuda.is_available() else 'cpu')
        )

    def predict(self, input_lamda, input_r):
        # Interpolate input spectrum to target wavelengths
        # input_lamda and input_r are lists or arrays
        
        # Ensure sorted
        sorted_indices = np.argsort(input_lamda)
        input_lamda = np.array(input_lamda)[sorted_indices]
        input_r = np.array(input_r)[sorted_indices]
        
        # Interpolate
        spectrum = np.interp(self.target_wavelengths, input_lamda, input_r)
        
        # Prepare tensor
        spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.net(spectrum_tensor)
            
            if isinstance(outputs, tuple):
                thickness, material_logits = outputs
                # Handle material logits
                if material_logits.dim() == 3:
                    material_indices = torch.argmax(material_logits, dim=-1)
                else:
                    material_indices = (material_logits > 0).long()
            else:
                thickness = outputs
                # Assume periodic materials if not predicted
                num_layers = thickness.shape[1]
                material_indices = (torch.arange(num_layers, device=self.device) % 2).unsqueeze(0).repeat(thickness.shape[0], 1)
            
        # Post-process
        # Denormalize thickness
        min_th = 20
        max_th = 200
        
        # Keep as tensor for TMM
        thickness_nm = thickness * (max_th - min_th) + min_th
        
        # Material indices
        # material_indices is already calculated above
        
        # Calculate produced spectrum
        R_calc, T_calc = self.tmm.compute_spectrum(thickness_nm.cpu(), material_indices.cpu())
        produced_spectrum = R_calc.cpu().numpy()[0]

        # Convert to numpy for layers list
        thickness_np = thickness_nm.cpu().numpy()[0]
        material_indices_np = material_indices.cpu().numpy()[0]
        
        layers = []
        current_z = 0.0
        for i in range(len(thickness_np)):
            th = float(thickness_np[i])
            mat_idx = material_indices_np[i]
            mat_name = self.materials[mat_idx]
            
            layer = {
                "name": f"Layer {i+1}",
                "material": mat_name,
                "thickness": round(th, 2),
                "ref_index": self.refractive_indices[mat_name],
                "start_z": round(current_z, 2),
                "end_z": round(current_z + th, 2),
                "color": self.colors[mat_name],
                # Add dummy values for fields expected by frontend if needed
                "lamda": 0, # Not applicable for layer
                "r_val": 0, # Not applicable
                "val": 0,
                "impedance": 0
            }
            layers.append(layer)
            current_z += th
            
        return {
            "layers": layers,
            "produced_spectrum": {
                "lamda": self.target_wavelengths.tolist(),
                "r": produced_spectrum.tolist()
            }
        }
