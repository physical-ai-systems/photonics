import torch
from torch.utils.data import Dataset
import numpy as np
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Methods.TransferMatrixMethod.PhotonicTransferMatrixFast import PhotonicTransferMatrixFast
from Methods.TransferMatrixMethod.Structure              import Structure, VectorizedStructure
from Methods.TransferMatrixMethod.Layer                  import Layer, MultiLayer
from Methods.TransferMatrixMethod.Wavelength             import WaveLength
from Materials.Materials                                 import Material
from Methods.PhysicalQuantity                            import PhysicalQuantity


class PhotonicDatasetTMMFast(Dataset):
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
                 units="nm",
                 unit_prefix=None,
                 thickness_range=(20, 200),  # in nm
                 thickness_steps=1,
                 thickness_unit_prefix=None,  # nanometers
                 thickness_units="nm",      # nanometers
                 batch_size=10,
                 dataset_size=10**6,
                 train_dataset_size=None,
                 test_dataset_downsize=1000,
                 test_mode=False,
                 device=None
                 ):
        
        self.num_layers = num_layers
        self.validate_num_layers(self.num_layers)

        self.ranges, self.steps, self.units, self.unit_prefix = ranges, steps, units, unit_prefix

        self.thickness_range, self.thickness_steps, self.thickness_unit_prefix, self.thickness_units = thickness_range, thickness_steps, thickness_unit_prefix, thickness_units

        self.test_mode = test_mode
        self.dataset_size = dataset_size if not test_mode else dataset_size // test_dataset_downsize
        self.train_dataset_size = train_dataset_size if train_dataset_size is not None else dataset_size
        self.batch_size = batch_size

        self.device = device if device is not None else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.wavelength = WaveLength(ranges=self.ranges, steps=self.steps, units=self.units, unit_prefix=self.unit_prefix)
        self.wavelength.to(self.device)
        self.wavelength.broadcast([1, 1, self.wavelength.shape[-1]])

        self.method = PhotonicTransferMatrixFast(device=self.device)
        self.method.device = self.device

        self.Materials = {
        "SiO2" : Material(self.wavelength, name="SiO2", refractive_index=1.4618),
        "Air" : Material(self.wavelength, name="Air",  refractive_index=1)  
        }
    
    def __len__(self):
        return self.dataset_size
    
    def validate_num_layers(self, num_layers):
        # log2(num_layers) should be an integer
        # check if num_layers is a power of 2
        # 1000000 & (1000000 - 1) == 0  -> False

        if not (num_layers > 0 and ((num_layers & (num_layers - 1)) == 0)):
            raise ValueError("num_layers should be a power of 2 (e.g., 2, 4, 8, 16, 32, ...)")
    
    def set_seed_for_index(self, idx):
        """
        Set random seeds for reproducibility based on dataset index.
        
        Parameters:
        -----------
        idx : int
            Dataset index
        """
        if not self.test_mode:
            np.random.seed(idx)
            torch.manual_seed(idx)
        else:
            seed_offset = self.train_dataset_size if self.train_dataset_size is not None else 0
            np.random.seed(seed_offset + idx)
            torch.manual_seed(seed_offset + idx)
    
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
        
        # self.set_seed_for_index(idx)

        # Create boundary layers using MultiLayer
        boundary_refractive_indices = torch.cat([
            torch.as_tensor(self.Materials["Air"].refractive_index),
            torch.as_tensor(self.Materials["SiO2"].refractive_index)
        ], dim=1) # Shape: (1, 2(number of layers), Wavelengths) -> in the future (B,2,Wavelengths)
        
        boundary_layers = MultiLayer(material=Material(self.wavelength, name="Boundary_Material", refractive_index=boundary_refractive_indices), thickness=None)

        # Create random layers thickness and material choices
        layer_thickness = torch.rand(self.batch_size, self.num_layers, device=self.device) * (self.thickness_range[1] - self.thickness_range[0]) + self.thickness_range[0]  # nm
        layer_thickness = torch.round(layer_thickness / self.thickness_steps) * self.thickness_steps  # round to nearest step
        layer_thickness = layer_thickness.clamp(self.thickness_range[0], self.thickness_range[1])  # ensure in range
        layer_thickness = layer_thickness.unsqueeze(-1).repeat(1, 1, self.wavelength.shape[-1])  # Expand to match wavelength dimension  
        

        material_choice = torch.randint(0, 2, (self.batch_size, self.num_layers), dtype=torch.long, device=self.device)
        refractive_indices = torch.empty(self.batch_size, self.num_layers, self.wavelength.shape[-1], device=self.device)

        material_list = list(self.Materials.keys())
        for i, mat_name in enumerate(material_list):
            mat_mask = (material_choice == i)
            num_true = torch.count_nonzero(mat_mask)
            mat_n = torch.as_tensor(self.Materials[mat_name].refractive_index[0,...]).repeat(num_true, 1)
            refractive_indices[mat_mask,...] = mat_n
        
        thickness = PhysicalQuantity(
            values=layer_thickness,
            units=self.thickness_units,
            unit_prefix=self.thickness_unit_prefix,
            name=f"Layer_Thickness"
        )

        material = Material(
            self.wavelength,
            name=f"Material",
            refractive_index=refractive_indices
        )
        layers = MultiLayer(material=material, thickness=thickness)
        
        R_calc, T_calc = self.method.Reflectance_from_layers(
            layers, 
            boundary_layers,
            theta_0=torch.tensor(0).to(self.device), 
            mode='TE'
        )
        thickness = thickness.values[...,0]  # Remove wavelength dimension for output
        thickness = (thickness - self.thickness_range[0]) / (self.thickness_range[1] - self.thickness_range[0])  # Normalize thickness to be between 0 and 1
        return {
            'R': R_calc,
            'T': T_calc, 
            'material_choice': material_choice,           
            'layer_thickness': thickness
        }

    def compute_spectrum(self, layer_thickness, material_choice):
        """
        Compute the spectrum from thickness and material indices.
        
        Parameters:
        -----------
        layer_thickness : torch.Tensor
            Shape (Batch, Num_Layers). Thickness in nanometers.
        material_choice : torch.Tensor
            Shape (Batch, Num_Layers). Material indices.
            
        Returns:
        --------
        R, T : torch.Tensor
            Reflectance and Transmission spectra.
        """
        # Check if the material_choice is binary
        if not torch.all((material_choice == 0) | (material_choice == 1)):
            # convert to binary by thresholding at 0.5
            material_choice = (material_choice > 0.5).long()
            
        batch_size = layer_thickness.shape[0]
        
        # Create boundary layers using MultiLayer
        boundary_refractive_indices = torch.cat([
            torch.as_tensor(self.Materials["Air"].refractive_index),
            torch.as_tensor(self.Materials["SiO2"].refractive_index)
        ], dim=1) 
        
        boundary_layers = MultiLayer(material=Material(self.wavelength, name="Boundary_Material", refractive_index=boundary_refractive_indices), thickness=None)

        # Expand thickness to match wavelength dimension
        layer_thickness_exp = layer_thickness.unsqueeze(-1).repeat(1, 1, self.wavelength.shape[-1])
        
        refractive_indices = torch.empty(batch_size, self.num_layers, self.wavelength.shape[-1], device=self.device)

        material_list = list(self.Materials.keys())
        for i, mat_name in enumerate(material_list):
            mat_mask = (material_choice == i)
            num_true = torch.count_nonzero(mat_mask)
            if num_true > 0:
                mat_n = torch.as_tensor(self.Materials[mat_name].refractive_index[0,...]).repeat(num_true, 1)
                refractive_indices[mat_mask,...] = mat_n
        
        thickness = PhysicalQuantity(
            values=layer_thickness_exp,
            units=self.thickness_units,
            unit_prefix=self.thickness_unit_prefix,
            name=f"Layer_Thickness"
        )

        material = Material(
            self.wavelength,
            name=f"Material",
            refractive_index=refractive_indices
        )
        layers = MultiLayer(material=material, thickness=thickness)
        
        R_calc, T_calc = self.method.Reflectance_from_layers(
            layers, 
            boundary_layers,
            theta_0=torch.tensor(0).to(self.device), 
            mode='TE'
        )
        
        return R_calc, T_calc
