import os
import sys
sys.path.append(os.getcwd())
import torch
import numpy as np
import matplotlib.pyplot as plt
from Methods.TransferMatrixMethod.PhotonicTransferMatrixFast import PhotonicTransferMatrixFast
from Methods.FDTD.FDTD_1D import run_fdtd_1d

class MockObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def test_fdtd_vs_tmm():
    lambda_min = 400e-9
    lambda_max = 800e-9
    wavelengths = torch.linspace(lambda_min, lambda_max, 500)
    
    num_layers = 10
    n_sio2 = 1.4618
    n_air = 1.0
    
    torch.manual_seed(42)
    thicknesses_raw = torch.abs(torch.randn(num_layers) * 20e-9 + 100e-9)
    
    dx = 5e-9
    thicknesses = (thicknesses_raw / dx).round() * dx
    
    refractive_indices = torch.tensor([n_sio2 if i % 2 == 0 else n_air for i in range(num_layers)])
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    wavelengths = wavelengths.to(device)
    thicknesses = thicknesses.to(device)
    refractive_indices = refractive_indices.to(device)
    
    batch_size = 1
    n_wl = len(wavelengths)
    
    n_internal = refractive_indices.view(batch_size, num_layers, 1).expand(batch_size, num_layers, n_wl).type(torch.complex128)
    d_internal = thicknesses.view(batch_size, num_layers, 1)
    
    k_vec = (2 * np.pi / wavelengths).view(batch_size, 1, n_wl).type(torch.complex128)
    
    mock_wavelength = MockObject(k=k_vec)
    mock_material_internal = MockObject(refractive_index=n_internal, wavelength=mock_wavelength)
    mock_layers = MockObject(thickness=d_internal, material=mock_material_internal)
    
    n_boundary = torch.ones((batch_size, 2, n_wl), dtype=torch.complex128, device=device)
    mock_material_boundary = MockObject(refractive_index=n_boundary)
    mock_boundary_layers = MockObject(material=mock_material_boundary)
    
    TMM = PhotonicTransferMatrixFast(device=device)
    R_tmm, T_tmm = TMM.Reflectance_from_layers(
        layers=mock_layers, 
        boundary_layers=mock_boundary_layers, 
        theta_0=torch.tensor(0.0, device=device)
    )
    
    tmm_trans = T_tmm.squeeze(0).real.cpu()

    center_wl = (lambda_min + lambda_max) / 2
    bandwidth = lambda_max - lambda_min
    
    fdtd_res = run_fdtd_1d(
        thicknesses, 
        refractive_indices,
        center_wavelength=center_wl,
        bandwidth=bandwidth + 100e-9,
        resolution=5e-9,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    fdtd_freqs = fdtd_res['frequencies']
    fdtd_trans = fdtd_res['transmission_spectrum']
    
    c = 299792458
    mask = fdtd_freqs > 0
    fdtd_lambda = c / fdtd_freqs[mask]
    fdtd_trans_masked = fdtd_trans[mask]
    
    sort_idx = torch.argsort(fdtd_lambda)
    fdtd_lambda_sorted = fdtd_lambda[sort_idx]
    fdtd_trans_sorted = fdtd_trans_masked[sort_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths.cpu().numpy()*1e9, tmm_trans.numpy(), label='TMM (Exact)', linewidth=2)
    plt.plot(fdtd_lambda_sorted.cpu().numpy()*1e9, fdtd_trans_sorted.cpu().numpy(), '--', label='FDTD', linewidth=2)
    plt.xlim(lambda_min*1e9, lambda_max*1e9)
    plt.ylim(0, 1.1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission')
    plt.title('Comparison: TMM vs FDTD')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('Methods/FDTD/vis/fdtd_vs_tmm.png')

if __name__ == "__main__":
    test_fdtd_vs_tmm()