import torch
import fdtd
import numpy as np
from typing import Dict, Optional, Tuple, Union

SPEED_OF_LIGHT = 299792458.0

class FDTD:
    
    def __init__(
        self,
        center_wavelength: float = 600e-9,
        bandwidth: float = 200e-9,
        resolution: float = None,
        padding: float = 1e-6,
        courant_number: float = 0.9,
        device: str = 'cuda'
    ):
        self.center_wavelength = center_wavelength
        self.bandwidth = bandwidth
        self.padding = padding
        self.courant_number = courant_number
        
        self.device = self._configure_backend(device)
        
        if resolution is None:
            lambda_min = center_wavelength - bandwidth / 2
            resolution = lambda_min / 20.0
        self.grid_spacing = resolution
        
        self._normalization_cache = {}

    def _configure_backend(self, device: str) -> torch.device:
        if device == 'cuda' and torch.cuda.is_available():
            fdtd.set_backend("torch.cuda.float32")
            return torch.device('cuda')
        fdtd.set_backend("torch.float32")
        return torch.device('cpu')

    def run(self, layer_thicknesses: torch.Tensor, layer_refractive_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        layer_thicknesses = layer_thicknesses.to(self.device)
        layer_refractive_indices = layer_refractive_indices.to(self.device)
        
        n_structure, cells_per_layer = self._discretize_structure(layer_thicknesses, layer_refractive_indices)
        
        total_thickness = float(torch.sum(layer_thicknesses))
        sim_length = total_thickness + 2 * self.padding
        Nx = int(sim_length / self.grid_spacing)
        
        max_n = float(torch.max(n_structure)) if len(n_structure) > 0 else 1.0
        optical_path = sim_length * max_n
        time_to_run_seconds = 3 * optical_path / SPEED_OF_LIGHT
        
        cache_key = (Nx, int(time_to_run_seconds / (self.grid_spacing / SPEED_OF_LIGHT)))
        
        if cache_key in self._normalization_cache:
            I0 = self._normalization_cache[cache_key]
        else:
            n_air = torch.ones_like(n_structure)
            res_norm = self._run_single_sim(n_air, Nx, time_to_run_seconds)
            I0 = res_norm['transmission_raw'] + 1e-20
            self._normalization_cache[cache_key] = I0
            
        res_device = self._run_single_sim(n_structure, Nx, time_to_run_seconds)
        
        return {
            'frequencies': res_device['frequencies'],
            'transmission_spectrum': res_device['transmission_raw'] / I0,
            'reflection_spectrum': res_device['reflection_raw'] / I0,
            'time_data': res_device['time_data'],
            'grid_stats': res_device['grid_stats'],
            'source_spectrum': I0
        }

    def _discretize_structure(self, thicknesses: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cells_per_layer = (thicknesses / self.grid_spacing).round().long()
        n_structure = torch.repeat_interleave(indices, cells_per_layer)
        return n_structure, cells_per_layer

    def _run_single_sim(self, n_profile: torch.Tensor, Nx: int, time_to_run: float) -> Dict[str, torch.Tensor]:
        
        grid = fdtd.Grid(
            shape=(Nx, 1, 1),
            grid_spacing=self.grid_spacing,
            permittivity=1.0,
            permeability=1.0,
            courant_number=self.courant_number
        )
        
        grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
        grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
        grid[:, 0, :] = fdtd.PeriodicBoundary(name="pbound_y")
        grid[:, :, 0] = fdtd.PeriodicBoundary(name="pbound_z")
        
        self._inject_material(grid, n_profile, Nx)
        
        self._setup_source(grid)
        det_refl, det_trans = self._setup_detectors(grid, Nx)
        
        grid.run(total_time=time_to_run)
        
        return self._process_results(grid, det_refl, det_trans)

    def _inject_material(self, grid: fdtd.Grid, n_profile: torch.Tensor, Nx: int):
        start_idx = int(self.padding / self.grid_spacing)
        end_idx = start_idx + len(n_profile)
        
        if end_idx > Nx:
            n_profile = n_profile[:Nx - start_idx]
            end_idx = Nx
            
        epsilon_grid = torch.ones((Nx, 1, 1, 3), device=self.device)
        epsilon_grid[start_idx:end_idx, 0, 0, :] = n_profile.view(-1, 1)**2
        grid.inverse_permittivity = 1.0 / epsilon_grid

    def _setup_source(self, grid: fdtd.Grid):
        f_center = SPEED_OF_LIGHT / self.center_wavelength
        f_bw = (SPEED_OF_LIGHT / (self.center_wavelength - self.bandwidth/2)) - \
               (SPEED_OF_LIGHT / (self.center_wavelength + self.bandwidth/2))
        
        cycles = max(1, int(f_center / f_bw))
        period_steps = int((1/f_center) / grid.time_step)
        
        source_pos_idx = int((self.padding * 0.3) / self.grid_spacing) + 10
        
        grid[source_pos_idx, 0, 0] = fdtd.PointSource(
            period=period_steps,
            amplitude=1.0,
            name="source",
            pulse=True,
            cycle=cycles
        )

    def _setup_detectors(self, grid: fdtd.Grid, Nx: int) -> Tuple[fdtd.LineDetector, fdtd.LineDetector]:
        refl_pos = int((self.padding * 0.8) / self.grid_spacing)
        trans_pos = Nx - int((self.padding * 0.2) / self.grid_spacing) - 10
        
        det_refl = fdtd.LineDetector(name="reflection")
        det_trans = fdtd.LineDetector(name="transmission")
        
        grid[refl_pos, :, :] = det_refl
        grid[trans_pos, :, :] = det_trans
        
        return det_refl, det_trans

    def _process_results(self, grid: fdtd.Grid, det_refl: fdtd.LineDetector, det_trans: fdtd.LineDetector) -> Dict:
        refl_t = torch.stack(det_refl.E)[..., 2].flatten()
        trans_t = torch.stack(det_trans.E)[..., 2].flatten()
        
        freqs = torch.fft.rfftfreq(len(refl_t), d=grid.time_step, device=self.device)
        
        return {
            'frequencies': freqs,
            'transmission_raw': torch.abs(torch.fft.rfft(trans_t))**2,
            'reflection_raw': torch.abs(torch.fft.rfft(refl_t))**2,
            'time_data': {'reflection': refl_t, 'transmission': trans_t, 'time_step': grid.time_step},
            'grid_stats': {'Nx': grid.shape[0], 'dx': grid.grid_spacing, 'dt': grid.time_step}
        }


def run_fdtd_1d(
    layer_thicknesses: torch.Tensor,
    layer_refractive_indices: torch.Tensor,
    center_wavelength: float = 600e-9,
    bandwidth: float = 200e-9,
    resolution: float = None,
    padding: float = 1e-6,
    courant_number: float = 0.9,
    device: str = 'cuda'
):
    solver = FDTD(
        center_wavelength=center_wavelength,
        bandwidth=bandwidth,
        resolution=resolution,
        padding=padding,
        courant_number=courant_number,
        device=device
    )
    return solver.run(layer_thicknesses, layer_refractive_indices)

if __name__ == "__main__":
    try:
        thick = torch.tensor([100e-9, 200e-9] * 5)
        n_idx = torch.tensor([1.5, 2.5] * 5)
        res = run_fdtd_1d(thick, n_idx, center_wavelength=600e-9)
        print("Simulation complete.")
        print(f"Transmission spectrum shape: {res['transmission_spectrum'].shape}")
    except Exception as e:
        print(f"Error: {e}")
