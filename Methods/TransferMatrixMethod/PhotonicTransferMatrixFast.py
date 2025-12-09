import torch
import numpy as np

class PhotonicTransferMatrixFast:
    def __init__(self, *args, **kwargs):
                 
        self.name       = 'PhotonicTransferMatrixFast'
        self.pi         = np.pi
        self.eps        = 1e-10
        self.device     = kwargs.get('device', torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.check_reshape = kwargs.get('check_reshape', False)

    def p_value(self, material, mode, theta):
        # This function gives p value
        if   mode == "TE":
            p = torch.cos(theta) * material.refractive_index
        elif mode == "TM":
            p = torch.cos(theta) / material.refractive_index
        else:
            raise ValueError("mode must be TE or TM")
        return p
    
    def k_wavevector(self, material):
        # This function gives k value
        k = material.wavelength.k
        return k
    
    def snells_law(self, n1, n2, theta1):
        # This function applies Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
        # Use clamp to avoid numerical issues with asin (domain: [-1, 1])
        sin_theta2 = (n1 / n2) * torch.sin(theta1)
        if not sin_theta2.is_complex():
            sin_theta2 = torch.clamp(sin_theta2, -1.0, 1.0)  # Ensure valid domain for asin
        theta2 = torch.asin(sin_theta2)      
        return theta2
    
    def transfer_matrix(self, layer, theta=0, mode='TE'):
        # This function gives the transfer matrix
        k0 = self.k_wavevector(layer.material)        
        p = self.p_value(layer.material, mode, theta)
        
        # Pre-compute common terms with safe operations
        cos_theta = torch.cos(theta)
        delta = layer.thickness * layer.material.refractive_index * cos_theta
        phase = k0 * delta
        
        # Compute trig functions once
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)
        
        # Use safe division to avoid NaN gradients
        # Add small epsilon to prevent division by zero
        inv_p = 1.0 / (p + self.eps * (p.abs() < self.eps).float())
        neg_1j_inv_p = -1j * inv_p
        neg_1j_p = -1j * p
        
        # Construct matrix directly using stack (more memory efficient)
        M = torch.stack([
            torch.stack([cos_phase, neg_1j_inv_p * sin_phase], dim=-1),
            torch.stack([neg_1j_p * sin_phase, cos_phase], dim=-1)
        ], dim=-2)
      
        return M
    
    def verify_reshape(self, M_original, num_checks=4):
        """
        Verify the reshape operation for pairwise matrix multiplication.
        
        Args:
            M_original: Original tensor to reshape
            num_checks: Number of verification checks to perform (default: 4)
        
        Returns:
            tuple: (is_valid, M_split) where is_valid is True if all checks pass
        
        Example:
            # For a tensor M with shape [batch, num_layers, wavelengths, 2, 2]
            # is_valid, M_split = self.verify_reshape(M)
            # 
            # This verifies:
            # M_split[0, 0, 0, :, :, :] == M[0, 0, :, :, :]  # True
            # M_split[0, 0, 1, :, :, :] == M[0, 1, :, :, :]  # True
            # M_split[0, 1, 0, :, :, :] == M[0, 2, :, :, :]  # True
            # M_split[0, 1, 1, :, :, :] == M[0, 3, :, :, :]  # True
        """
        M_split = M_original.contiguous().view(*M_original.shape[0:1], M_original.shape[1]//2, 2, *M_original.shape[2:-2], 2, 2)
        checks = []
        
        for i in range(min(num_checks, M_original.shape[1])):
            pair_idx = i // 2
            sub_idx = i % 2
            checks.append(torch.all(M_split[0, pair_idx, sub_idx, :, :, :] == M_original[0, i, :, :, :]))
        
        return all(checks), M_split
    

    def Reflectance(self, boundary_layers, transfer_matrix, boundary_theta, mode='TE'): 
    # This function gives the reflectance of the structure

        p = self.p_value(boundary_layers.material, mode, boundary_theta)
        M = transfer_matrix

        denominator = (M[...,0,0] + M[...,0,1] * p[:,1,...]) * p[:,0,...] + (M[...,1,0] + M[...,1,1] * p[:,1,...] )

        numerator_r = (M[...,0,0] + M[...,0,1] * p[:,1,...]) * p[:,0,...] - (M[...,1,0] + M[...,1,1] * p[:,1,...] )
        
        r = numerator_r / denominator
        R = torch.abs(r) ** 2

        t_numerator = 2 * p[:,0,...]
        t = t_numerator / denominator
        T = (p[:,1,...] / p[:,0,...]) * torch.abs(t) ** 2

        return R, T
    

    def Reflectance_from_layers(self, layers, boundary_layers, theta_0=torch.tensor(0), mode='TE'):
        """
        Calculate the reflectance and transmittance of a multilayer structure using the Transfer Matrix Method.
        
        This implementation uses a fast pairwise reduction algorithm for matrix multiplication,
        achieving O(log N) depth complexity where N is the number of layers. This is particularly
        efficient for GPUs.

        Args:
            layers: Object containing material properties (refractive index) and thickness for the internal layers.
            boundary_layers: Object containing material properties for the input (incidence) and output (substrate) media.
            theta_0 (torch.Tensor, optional): Incident angle in radians. Defaults to 0 (normal incidence).
            mode (str, optional): Polarization mode, either 'TE' (Transverse Electric) or 'TM' (Transverse Magnetic). Defaults to 'TE'.

        Returns:
            tuple: A tuple (R, T) containing:
                - R (torch.Tensor): Reflectance of the structure.
                - T (torch.Tensor): Transmittance of the structure.
        """
        
        theta = self.snells_law(boundary_layers.material.refractive_index[:,0,...].unsqueeze(1),layers.material.refractive_index, theta_0)
        boundary_theta = torch.cat([theta_0.broadcast_to(boundary_layers.material.refractive_index[:,0,...].unsqueeze(1).shape), 
                        self.snells_law(boundary_layers.material.refractive_index[:,0,...].unsqueeze(1),boundary_layers.material.refractive_index[:,1,...].unsqueeze(1), theta_0)
                                    ], dim=1)
        # Build per-layer transfer matrices for internal layers (exclude boundaries)
        # M = [ batch, layers, wavelengths, 2, 2 ]
        M = self.transfer_matrix(layers, theta, mode)

        # Pairwise reduction to compute M with ~log2(N) matmul depth
        # Example for 8 layers: M = (m1 m2)(m3 m4)(m5 m6)(m7 m8)
        #  M = (m1 m3 m5 m7) @ (m2 m4 m6 m8) -> (m12 m34 m56 m78) -> ...
        # M is of shape [batch, num_layers, wavelengths, 2, 2]

        if self.check_reshape:
            is_valid, M_split = self.verify_reshape(M)
            if not is_valid:
                raise ValueError("Reshape verification failed in Reflectance_from_layers.")
            M = M_split

        while M.shape[1] > 1:
            if M.shape[1] % 2 == 1:
                padding_shape = list(M.shape)
                padding_shape[1] = 1
                eye = torch.eye(2, dtype=torch.complex128, device=self.device).view([1] * (len(M.shape) - 2) + [2, 2])
                M = torch.cat([M, eye.expand(padding_shape)], dim=1)
            M = M.contiguous().view(*M.shape[0:1], M.shape[1]//2, 2, *M.shape[2:-2], 2, 2) # [batch, layers//2, 2, wavelengths, 2, 2]
            M = torch.matmul(M[...,0,:,:,:], M[...,1,:,:,:])
        M = M.squeeze(1).contiguous()  # Remove layer dimension

        R, T = self.Reflectance(boundary_layers, M, boundary_theta, mode)
        return R, T
    
     


    

    