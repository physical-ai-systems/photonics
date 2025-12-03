import torch
from torch.utils.data import Dataset
import numpy as np
import os
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Methods.TransferMatrixMethod.Structure import Structure
from Methods.TransferMatrixMethod.Layer import Layer
from Methods.TransferMatrixMethod.Wavelength import WaveLength
from Materials.Materials import Material
from Methods.PhysicalQuantity import PhysicalQuantity
from Materials.refractiveindex_sqlite.refractivesqlite.dboperations import Database

class MaterialDataset(Dataset):
    def __init__(self,
                 num_layers=20,
                 num_materials=3,
                 ranges=(400, 700),
                 steps=1,
                 units="m",
                 unit_prefix="n",
                 thickness_range=(20, 200),
                 thickness_steps=1,
                 thickness_unit_prefix="n",
                 thickness_units="m",
                 batch_size=10,
                 dataset_size=10**6,
                 train_dataset_size=None,
                 test_dataset_downsize=100,
                 test_mode=False,
                 device=None,
                 db_path=None):
        
        if db_path is None:
            # Resolve default path: ../Materials/refractiveindex_sqlite/refractive.db
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            db_path = os.path.join(project_root, 'Materials', 'refractiveindex_sqlite', 'refractive.db')

        self.num_layers = num_layers
        self.validate_num_layers(self.num_layers)

        self.dataset_size = dataset_size
        self.num_materials = num_materials
        self.ranges = ranges
        self.steps = steps
        self.units = units
        self.unit_prefix = unit_prefix
        self.thickness_range = thickness_range
        self.thickness_steps = thickness_steps
        self.thickness_unit_prefix = thickness_unit_prefix
        self.thickness_units = thickness_units
        self.batch_size = batch_size
        self.test_mode = test_mode
        self.device = device if device is not None else torch.device('cpu')
        
        self.wavelength = WaveLength(
            ranges=self.ranges,
            steps=self.steps,
            units=self.units,
            unit_prefix=self.unit_prefix
        )
        self.wavelength.to(self.device)
        self.wavelength.broadcast([self.batch_size, self.wavelength.shape[-1]])
        
        self.method = PhotonicTransferMatrix()
        
        self.wavelength_nm = torch.arange(ranges[0], ranges[1] + steps/100, steps, dtype=torch.float32).to(self.device)

        self.db = Database(db_path)
        min_um = ranges[0] / 1000.0
        max_um = ranges[1] / 1000.0
        
        query = f"SELECT pageid FROM pages WHERE rangeMin <= {min_um} AND rangeMax >= {max_um}"
        results = self.db.search_custom(query)
        
        if not results:
             raise RuntimeError(f"No materials found in range {ranges} nm")

        raw_ids = [r[0] for r in results]
        print(f"Found {len(raw_ids)} materials covering the range.")
        
        self.material_cache = {}
        wl_np = self.wavelength_nm.cpu().numpy()
        
        for pid in raw_ids:
            try:
                mat = self.db.get_material(pid)
                if mat is None: continue
                
                if mat.has_refractive():
                    n = mat.get_refractiveindex(wl_np.copy())
                else:
                    continue 
                    
                if mat.has_extinction():
                    k = mat.get_extinctioncoefficient(wl_np.copy())
                else:
                    k = np.zeros_like(wl_np)
                
                if isinstance(n, (float, int)):
                    n = np.full_like(wl_np, n)
                if isinstance(k, (float, int)):
                    k = np.full_like(wl_np, k)

                ri_complex = torch.from_numpy(n).float() + 1j * torch.from_numpy(k).float()
                self.material_cache[pid] = ri_complex.to(self.device)
                
            except Exception as e:
                print(f"Skipping material {pid}: {e}")
        
        self.valid_ids = list(self.material_cache.keys())
        print(f"Cached {len(self.valid_ids)} valid materials.")

        if len(self.valid_ids) < self.num_materials:
             print("Warning: Not enough valid materials. Adjusting num_materials.")
             self.num_materials = len(self.valid_ids)

        if not self.valid_ids:
            raise RuntimeError("No valid materials loaded.")

    def validate_num_layers(self, num_layers):
        if not (num_layers > 0 and ((num_layers & (num_layers - 1)) == 0)):
            raise ValueError("num_layers should be a power of 2")

    def __len__(self):
        return self.dataset_size

    @torch.no_grad()
    def __getitem__(self, idx):
        if not self.test_mode:
            seed = idx
            torch.manual_seed(seed)
            np.random.seed(seed)

        chosen_ids = np.random.choice(self.valid_ids, self.num_materials, replace=False)
        
        ri_list = [self.material_cache[pid] for pid in chosen_ids]
        ri_stack = torch.stack(ri_list) 
        
        material_choice = torch.randint(0, self.num_materials + 1, (self.batch_size, self.num_layers), dtype=torch.long, device=self.device)
        
        refractive_indices = torch.ones(self.batch_size, self.num_layers, self.wavelength.shape[-1], dtype=torch.complex64, device=self.device)
        
        for k in range(1, self.num_materials + 1):
            mask = (material_choice == k) 
            if mask.any():
                ri_vals = ri_stack[k-1].unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.num_layers, -1)
                refractive_indices[mask] = ri_vals[mask]

        t_raw = torch.rand(self.batch_size, self.num_layers, device=self.device)
        t_scaled = t_raw * (self.thickness_range[1] - self.thickness_range[0]) + self.thickness_range[0]
        t_rounded = torch.round(t_scaled / self.thickness_steps) * self.thickness_steps
        layer_thickness = t_rounded.clamp(self.thickness_range[0], self.thickness_range[1])
        
        layer_thickness_exp = layer_thickness.unsqueeze(-1).repeat(1, 1, self.wavelength.shape[-1])

        air_boundary = Material(self.wavelength, name="Air", refractive_index=1.0)
        substrate = Material(self.wavelength, name="SiO2", refractive_index=1.4618)

        thickness_pq = PhysicalQuantity(
            values=layer_thickness_exp,
            units=self.thickness_units,
            unit_prefix=self.thickness_unit_prefix,
            name="Layer_Thickness"
        )
        
        layers = []
        for i in range(self.num_layers):
            mat = Material(
                self.wavelength, 
                name=f"L{i}", 
                refractive_index=refractive_indices[:, i, :]
            )
            layers.append(Layer(material=mat, thickness=thickness_pq.values[:, i, :]))
            
        struct = Structure(
            layers=[Layer(air_boundary)] + layers + [Layer(substrate)], 
            layers_parameters={'method': 'multi_layer'}
        )
        
        R, T = self.method.Reflectance_from_layers(struct.layers, theta_0=0, mode='TE')
        
        min_th, max_th = self.thickness_range
        norm_thickness = (layer_thickness - min_th) / (max_th - min_th)
        
        return {
            'R': R.detach().float(),
            'T': T.detach().float(),
            'material_choice': material_choice,
            'layer_thickness': norm_thickness.detach().float()
        }
