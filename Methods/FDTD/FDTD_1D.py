import torch
import fdtd
import numpy as np

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
    if device == 'cuda' and torch.cuda.is_available():
        fdtd.set_backend("torch.cuda.float32")
        device_torch = torch.device('cuda')
    else:
        fdtd.set_backend("torch.float32")
        device_torch = torch.device('cpu')

    lambda_min = center_wavelength - bandwidth / 2
    if resolution is None:
        resolution = lambda_min / 20.0
    
    grid_spacing = resolution

    layer_thicknesses = layer_thicknesses.to(device_torch)
    layer_refractive_indices = layer_refractive_indices.to(device_torch)

    total_thickness = float(torch.sum(layer_thicknesses))
    sim_length = total_thickness + 2 * padding
    Nx = int(sim_length / grid_spacing)
    
    cells_per_layer = (layer_thicknesses / grid_spacing).round().long()
    n_structure = torch.repeat_interleave(layer_refractive_indices, cells_per_layer)
    
    c = 299792458
    max_n = float(torch.max(n_structure)) if len(n_structure) > 0 else 1.0
    optical_path = sim_length * max_n
    time_to_run = 3 * optical_path / c 

    print("Running FDTD Normalization (Air)...")
    n_air = torch.ones_like(n_structure)
    res_norm = _run_simulation_core(
        n_profile=n_air,
        Nx=Nx,
        grid_spacing=grid_spacing,
        padding=padding,
        center_wavelength=center_wavelength,
        bandwidth=bandwidth,
        courant_number=courant_number,
        device_torch=device_torch,
        time_to_run=time_to_run
    )
    
    I0 = res_norm['transmission_raw'] + 1e-20 
    
    print("Running FDTD Device Simulation...")
    res_device = _run_simulation_core(
        n_profile=n_structure,
        Nx=Nx,
        grid_spacing=grid_spacing,
        padding=padding,
        center_wavelength=center_wavelength,
        bandwidth=bandwidth,
        courant_number=courant_number,
        device_torch=device_torch,
        time_to_run=time_to_run
    )
    
    T = res_device['transmission_raw'] / I0
    R = res_device['reflection_raw'] / I0
    
    results = {
        'frequencies': res_device['frequencies'],
        'transmission_spectrum': T,
        'reflection_spectrum': R,
        'time_data': res_device['time_data'],
        'grid_stats': res_device['grid_stats'],
        'source_spectrum': I0
    }
    
    return results

def _run_simulation_core(
    n_profile, Nx, grid_spacing, padding, 
    center_wavelength, bandwidth, courant_number, device_torch, time_to_run
):
    """
    Helper function to run a single FDTD simulation with a given refractive index profile.
    """
    grid = fdtd.Grid(
        shape=(Nx, 1, 1),
        grid_spacing=grid_spacing,
        permittivity=1.0,
        permeability=1.0,
        courant_number=courant_number
    )
    
    grid[0:10, :, :] = fdtd.PML(name="pml_xlow")
    grid[-10:, :, :] = fdtd.PML(name="pml_xhigh")
    grid[:, 0, :] = fdtd.PeriodicBoundary(name="pbound_y")
    grid[:, :, 0] = fdtd.PeriodicBoundary(name="pbound_z")
    
    start_idx = int(padding / grid_spacing)
    end_idx = start_idx + len(n_profile)
    
    if end_idx > Nx:
        n_profile = n_profile[:Nx - start_idx]
        end_idx = Nx
        
    epsilon_grid = torch.ones((Nx, 1, 1, 3), device=device_torch)
    epsilon_grid[start_idx:end_idx, 0, 0, :] = n_profile.view(-1, 1)**2
    grid.inverse_permittivity = 1.0 / epsilon_grid
    
    c = 299792458
    f_center = c / center_wavelength
    f_bandwidth = (c / (center_wavelength - bandwidth/2)) - (c / (center_wavelength + bandwidth/2))
    cycles = int(f_center / f_bandwidth)
    if cycles < 1: cycles = 1
    
    source_pos_idx = int((padding * 0.3) / grid_spacing) + 10
    
    period_steps = int((1/f_center) / grid.time_step)
    
    source = fdtd.PointSource(
        period=period_steps,
        amplitude=1.0,
        name="source",
        pulse=True,
        cycle=cycles
    )
    grid[source_pos_idx, 0, 0] = source
    
    refl_pos_idx = int((padding * 0.8) / grid_spacing)
    det_reflection = fdtd.LineDetector(name="reflection")
    grid[refl_pos_idx, :, :] = det_reflection
    
    trans_pos_idx = Nx - int((padding * 0.2) / grid_spacing) - 10
    det_transmission = fdtd.LineDetector(name="transmission")
    grid[trans_pos_idx, :, :] = det_transmission
    
    grid.run(total_time=time_to_run)
    
    refl_tensor = torch.stack(det_reflection.E) 
    trans_tensor = torch.stack(det_transmission.E)
    
    refl_t = refl_tensor[..., 2].flatten()
    trans_t = trans_tensor[..., 2].flatten()
    
    freqs = torch.fft.rfftfreq(len(refl_t), d=grid.time_step, device=refl_t.device)
    refl_f = torch.fft.rfft(refl_t)
    trans_f = torch.fft.rfft(trans_t)
    
    T_raw = torch.abs(trans_f)**2
    R_raw = torch.abs(refl_f)**2
    
    return {
        'frequencies': freqs,
        'transmission_raw': T_raw,
        'reflection_raw': R_raw,
        'time_data': {'reflection': refl_t, 'transmission': trans_t, 'time_step': grid.time_step},
        'grid_stats': {'Nx': Nx, 'dx': grid_spacing, 'dt': grid.time_step}
    }

if __name__ == "__main__":
    try:
        thick = torch.tensor([100e-9, 200e-9] * 5)
        n_idx = torch.tensor([1.5, 2.5] * 5)
        res = run_fdtd_1d(thick, n_idx, center_wavelength=600e-9)
        print("Simulation complete.")
        print(f"Transmission spectrum shape: {res['transmission_spectrum'].shape}")
    except Exception as e:
        print(f"Error: {e}")