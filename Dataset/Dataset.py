import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.interpolate import CubicSpline

from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Methods.TransferMatrixMethod.Structure              import Structure
from Methods.TransferMatrixMethod.Layer                  import Layer
from Methods.TransferMatrixMethod.Wavelength             import WaveLength
from Materials.Materials                                 import Material
from Methods.PhysicalQuantity                            import PhysicalQuantity


class PhotonicDataset(Dataset):
    """
    PyTorch Dataset for photonic structures that generates data on-the-fly.
    
    Parameters:
    -----------
    num_layers : int
        Number of layers in the photonic structure (default: 20)
    ranges : tuple
        Wavelength range in nm (min_nm, max_nm)
    steps : int
        Step size in nm for fixed wavelength points
    units : str
        Units for wavelength (default: 'm' for meters)
    unit_prefix : str
        Unit prefix for wavelength (default: 'n' for nanometers)
    dataset_size : int
        Number of samples in the dataset (default: 10^6)    
    """
    
    @torch.no_grad()
    def __init__(self,
                 num_layers=20,
                 ranges=(400, 700),
                 steps=1,
                 units="m",
                 unit_prefix="n",
                 thickness_range=(20, 200),  # in nm
                 thickness_steps=1,
                 thickness_unit_prefix="n",  # nanometers
                 thickness_units="m",      # meters
                 batch_size=10,
                 dataset_size=10**6,
                 ):
        
        self.num_layers = num_layers
        self.ranges = ranges  # (min_nm, max_nm)
        self.steps = steps    # step in nm
        self.units = units
        self.unit_prefix = unit_prefix
        self.thickness_range = thickness_range
        self.thickness_steps = thickness_steps
        self.thickness_unit_prefix = thickness_unit_prefix
        self.thickness_units = thickness_units
        self.dataset_size = dataset_size
        self.batch_size = batch_size

        self.wavelength = WaveLength(
            ranges=self.ranges,
            steps=self.steps,
            units=self.units,
            unit_prefix=self.unit_prefix
        )

        self.wavelength.broadcast([self.batch_size, self.wavelength.shape[-1]])

        # Initialize the transfer matrix method
        self.method = PhotonicTransferMatrix()

        self.Materials = {
        "SiO2" : Material(self.wavelength, name="SiO2", refractive_index=1.4618),
        "Air" : Material(self.wavelength, name="Air",  refractive_index=1)  
        }
    
    def __len__(self):
        return self.dataset_size
    
    @torch.no_grad()
    def __getitem__(self, idx):
        """
        Generate a single sample on-the-fly.
        
        Returns:
        --------
        dict with keys:
            'wavelength': torch.Tensor of shape (N,) - wavelength values in nm
            'R': torch.Tensor of shape (N,) - interpolated reflectance values
            'T': torch.Tensor of shape (N,) - interpolated transmission values
            'R_fixed': torch.Tensor of shape (10,) - reflectance at fixed wavelengths
            'material_choice': int - 0 for Air, 1 for Silicon
            'layer_thickness': float - thickness of the middle layer in nm
        """
        
        # Seed for reproducibility of this specific index
        np.random.seed(idx)
        torch.manual_seed(idx)

        # Randomly choose material (0: Air, 1: Silicon) using PyTorch
        material_choice = torch.randint(0, 2, (self.batch_size, self.num_layers), dtype=torch.long)

        # Generate random layer thicknesses for batch
        layer_thickness = torch.rand(self.batch_size, self.num_layers) * (self.thickness_range[1] - self.thickness_range[0]) + self.thickness_range[0]  # nm
        layer_thickness = torch.round(layer_thickness / self.thickness_steps) * self.thickness_steps  # round to nearest step
        layer_thickness = layer_thickness.clamp(self.thickness_range[0], self.thickness_range[1])  # ensure in range
        layer_thickness = layer_thickness.unsqueeze(-1).repeat(1, 1, self.wavelength.shape[-1])  # Expand to match wavelength dimension  
        
        refractive_indices = torch.empty(self.batch_size, self.num_layers, self.wavelength.shape[-1])

        # Vectorized assignment using boolean indexing
        material_list = list(self.Materials.keys())
        for i, mat_name in enumerate(material_list):
            mat_mask = (material_choice == i)
            num_true = torch.count_nonzero(mat_mask)
            mat_n = torch.tensor(self.Materials[mat_name].refractive_index[0,...]).unsqueeze(0).repeat(num_true, 1)
            refractive_indices[mat_mask,...] = mat_n
        
        air_boundary = self.Materials["Air"]
        substrate = self.Materials["SiO2"]

        thickness = PhysicalQuantity(
            values=layer_thickness,
            units=self.thickness_units,
            unit_prefix=self.thickness_unit_prefix,
            name=f"Layer_Thickness"
        )

        layers = []

        for layer_idx in range(self.num_layers):
            refractive_indices_layer = refractive_indices[:, layer_idx, :]
            thickness_layer = thickness.values[:, layer_idx, :]
            material = Material(
                self.wavelength,
                name=f"Layer_{layer_idx}_Material",
                refractive_index=refractive_indices_layer
            )
            layer = Layer(material=material, thickness=thickness_layer)
            layers.append(layer)
        # Create structure
        structure = Structure(layers=[Layer(air_boundary)] + layers + [Layer(substrate)],
                              layers_parameters={'method': 'multi_layer'})
        
        # Calculate reflectance and transmission
        R_calc, T_calc = self.method.Reflectance_from_layers(
            structure.layers, 
            theta_0=0, 
            mode='TE'
        )
        
        return {
            'R': torch.tensor(R_calc, dtype=torch.float32),
            'T': torch.tensor(T_calc, dtype=torch.float32), 
            'material_choice': material_choice,           
            'layer_thickness': torch.tensor(layer_thickness, dtype=torch.float32)
        } 