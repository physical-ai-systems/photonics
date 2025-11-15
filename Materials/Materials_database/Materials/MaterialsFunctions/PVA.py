import torch


def refractive_index_PVA(wavelength, gamma_ray_dose):

    """
        Reference:
        Optical and Quantum Electronics (2022) 54:755
        https://doi.org/10.1007/s11082-022-04189-3
        1 3
        One‑dimentional periodic structure infiltrated by (PVA/
        CV + CF)‑polymer for high‑performance sensitivity
        Fatma A. Sayed1 · Hussein A. Elsayed1 · Ahmed Mehaney1 · M. F. Eissa1 ·
        Arafa H. Aly1
    """

  # Coefficients dictionary
    coefficients = torch.tensor([
        [1.7479e-7, -0.00025167, 0.11796, -15.571],
        [1.5689e-7, -0.00022845, 0.1083, -14.282],
        [1.3002e-7, -0.00019042, 0.090693, -11.606],
        [1.2167e-7, -0.00017806, 0.084739, -10.663],
        [1.0444e-7, -0.00015298, 0.07293, -8.87],
        [1.013e-7, -0.00014864, 0.070913, -8.5343],
        [9.754e-8, -0.00014305, 0.068283, -8.1388],
        [9.6042e-8, -0.00014056, 0.067049, -7.9382]
    ])
    
    # Valid gamma-ray doses
    valid_doses = torch.tensor([0, 10, 20, 30, 40, 50, 60, 70])
    
    # Ensure inputs are tensors
    gamma_ray_dose = torch.as_tensor(gamma_ray_dose)
    wavelength = torch.as_tensor(wavelength).unsqueeze(0).repeat(gamma_ray_dose.shape[0], 1)
    
    # Find the indices of the nearest valid doses
    _, dose_indices = torch.abs(gamma_ray_dose.unsqueeze(-1) - valid_doses).min(dim=-1)
    
    # Select the corresponding coefficients
    selected_coefficients = coefficients[dose_indices]
    
    # Unpack coefficients
    A, B, C, D = selected_coefficients.unbind(-1)

    A, B, C, D = A.unsqueeze(-1), B.unsqueeze(-1), C.unsqueeze(-1), D.unsqueeze(-1)

    
    # Calculate the refractive index using the equation n(D) = Aλ³ + Bλ² + Cλ + D
    n = A * wavelength**3 + B * wavelength**2 + C * wavelength + D
    
    return n