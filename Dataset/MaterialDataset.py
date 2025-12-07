import torch
from torch.utils.data import Dataset
import numpy as np
import sqlite3
import os
from Materials.refractiveindex_sqlite.refractivesqlite import dboperations as refractiveindex_database
from Materials.refractiveindex_sqlite.refractivesqlite.material import NoExtinctionCoefficient
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Methods.TransferMatrixMethod.Structure import Structure
from Methods.TransferMatrixMethod.Layer import Layer
from Methods.TransferMatrixMethod.Wavelength import WaveLength
from Materials.Materials import Material
from Methods.PhysicalQuantity import PhysicalQuantity

class SqliteMaterialDataset(Dataset):
    def __init__(self,
                 structure_layers=20,
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
                 db_path=None,
                 spectrum_len=None):
        
        self.structure_layers = structure_layers
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
        self.spectrum_len = spectrum_len
        
        self.wavelength = WaveLength(
            ranges=self.ranges,
            steps=self.steps,
            units=self.units,
            unit_prefix=self.unit_prefix
        )
        self.wavelength.to(self.device)
        self.wavelength.broadcast([self.batch_size, self.wavelength.shape[-1]])
        
        self.method = PhotonicTransferMatrix()

        self.wavelength_nm = torch.arange(ranges[0], ranges[1] + steps/100, steps, dtype=torch.float32)
        
        if db_path is None:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_path, 'Materials', 'refractiveindex_sqlite', 'refractive.db')
        else:
            self.db_path = db_path
            
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Refractive database not found at {self.db_path}")

        self.db_helper = refractiveindex_database.Database(self.db_path)
        self.valid_materials = self._scan_database()

    def _scan_database(self):
        range_min_um = self.ranges[0] / 1000.0
        range_max_um = self.ranges[1] / 1000.0
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        query = """
        SELECT pageid, shelf, book, page, filepath 
        FROM pages 
        WHERE rangeMin <= ? 
          AND rangeMax >= ? 
          AND hasrefractive = 1
        """
        c.execute(query, (range_min_um, range_max_um))
        results = c.fetchall()
        conn.close()
        materials = []
        for row in results:
            materials.append({
                'pageid': row[0],
                'shelf': row[1],
                'book': row[2],
                'page': row[3],
                'filepath': row[4]
            })
        return materials

    def __len__(self):
        return len(self.valid_materials)

    @torch.no_grad()
    def __getitem__(self, idx):
        mat_info = self.valid_materials[idx]
        pageid = mat_info['pageid']
        material_db = self.db_helper.get_material(pageid)
        
        if material_db is None:
            return self.__getitem__((idx + 1) % len(self))

        try:
            n_values = [float(material_db.get_refractiveindex(wl.item())) for wl in self.wavelength_nm]
        except Exception:
            n_values = torch.ones_like(self.wavelength_nm)
        
        n_tensor = torch.tensor(n_values, dtype=torch.float32)

        try:
            k_values = [float(material_db.get_extinctioncoefficient(wl.item())) for wl in self.wavelength_nm]
            k_tensor = torch.tensor(k_values, dtype=torch.float32)
        except (NoExtinctionCoefficient, Exception):
            k_tensor = torch.zeros_like(n_tensor)
            
        ri_tensor = torch.complex(n_tensor, k_tensor).to(self.device)
        
        if not self.test_mode:
            np.random.seed(idx)
            torch.manual_seed(idx)
        
        material_choice = torch.randint(0, 2, (self.batch_size, self.structure_layers), dtype=torch.long, device=self.device)
        
        layer_thickness = torch.rand(self.batch_size, self.structure_layers, device=self.device) * (self.thickness_range[1] - self.thickness_range[0]) + self.thickness_range[0]
        layer_thickness = torch.round(layer_thickness / self.thickness_steps) * self.thickness_steps
        layer_thickness = layer_thickness.clamp(self.thickness_range[0], self.thickness_range[1])
        
        layer_thickness_exp = layer_thickness.unsqueeze(-1).repeat(1, 1, self.wavelength.shape[-1])
        
        refractive_indices = torch.ones(self.batch_size, self.structure_layers, self.wavelength.shape[-1], dtype=torch.complex64, device=self.device)
        
        mat_mask = (material_choice == 1)
        ri_expanded = ri_tensor.unsqueeze(0).unsqueeze(0).expand(self.batch_size, self.structure_layers, -1)
        refractive_indices[mat_mask] = ri_expanded[mat_mask]
        
        air_boundary = Material(self.wavelength, name="Air", refractive_index=1)
        substrate = Material(self.wavelength, name="SiO2", refractive_index=1.4618)
        
        thickness = PhysicalQuantity(
            values=layer_thickness_exp,
            units=self.thickness_units,
            unit_prefix=self.thickness_unit_prefix,
            name="Layer_Thickness"
        )
        
        layers = []
        for layer_idx in range(self.structure_layers):
            refractive_indices_layer = refractive_indices[:, layer_idx, :]
            thickness_layer = thickness.values[:, layer_idx, :]
            material = Material(
                self.wavelength,
                name=f"Layer_{layer_idx}_Material",
                refractive_index=refractive_indices_layer
            )
            layer = Layer(material=material, thickness=thickness_layer)
            layers.append(layer)
            
        structure = Structure(layers=[Layer(air_boundary)] + layers + [Layer(substrate)],
                              layers_parameters={'method': 'multi_layer'})
        
        R_calc, T_calc = self.method.Reflectance_from_layers(
            structure.layers, 
            theta_0=0, 
            mode='TE'
        )
        
        def to_float_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.detach().float()
            return torch.tensor(np.array(x).copy(), dtype=torch.float32, device=self.device)

        min_th, max_th = self.thickness_range
        target_thickness = (layer_thickness[..., 0] - min_th) / (max_th - min_th)

        return {
            'R': to_float_tensor(R_calc),
            'T': to_float_tensor(T_calc), 
            'material_choice': material_choice,           
            'layer_thickness': to_float_tensor(target_thickness),
            'name': f"{mat_info['shelf']}::{mat_info['book']}::{mat_info['page']}"
        }