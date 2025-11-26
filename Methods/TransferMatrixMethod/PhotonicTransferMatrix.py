import torch
import numpy as np
class PhotonicTransferMatrix:
    def __init__(self, *args, **kwargs):
                 
        self.name       = 'PhotonicTransferMatrix'
        self.pi         = np.pi

    def p_value(self, material, mode, theta):
        # This function gives p value
        device = material.refractive_index.device
        theta = torch.as_tensor(theta, dtype=torch.float64, device=device)
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
        device = n1.device if isinstance(n1, torch.Tensor) else (n2.device if isinstance(n2, torch.Tensor) else None)
        
        theta1 = torch.as_tensor(theta1, dtype=torch.float64, device=device)
        n1 = torch.as_tensor(n1, dtype=torch.float64, device=device)
        n2 = torch.as_tensor(n2, dtype=torch.float64, device=device)
        
        sin_theta2 = (n1 / n2) * torch.sin(theta1)
        theta2 = torch.asin(sin_theta2)
        
        return theta2
    
    def transfer_matrix(self, layer, theta=0, mode='TE'):
        # This function gives the transfer matrix
        k0 = self.k_wavevector(layer.material)
        device = k0.device
        
        theta = torch.as_tensor(theta, dtype=torch.float64, device=device)
        p = self.p_value(layer.material,mode, theta)

        
        M = torch.zeros((*k0.shape,2,2), dtype = torch.complex128, device=device)
        delta = layer.thickness * layer.material.refractive_index * torch.cos(theta)
        
        M[...,0,0] = torch.cos(k0 * delta)
        M[...,0,1] = -(  1j / p ) * torch.sin(k0 * delta)
        M[...,1,0] = ( -1j * p ) * torch.sin(k0 * delta) 
        M[...,1,1] = torch.cos(k0 * delta)
      
        return M
    

    def Reflectance(self, boundary_layers, transfer_matrix, theta, mode='TE'): 
    # This function gives the reflectance of the structure

        assert len(boundary_layers) == 2 , "boundary_layers must be a list of two layers" 
        assert mode in ["TE","TM"], "mode must be TE or TM"
        
        device = transfer_matrix.device
        theta_0 = torch.as_tensor(theta[0], dtype=torch.float64, device=device)
        theta_f = torch.as_tensor(theta[-1], dtype=torch.float64, device=device)

        jo = self.p_value(boundary_layers[0].material, mode, theta_0)
        js = self.p_value(boundary_layers[1].material, mode, theta_f)

        M = transfer_matrix

        denominator = (M[...,0,0] + M[...,0,1] * js) * jo + (M[...,1,0] + M[...,1,1] * js )

        numerator_r = (M[...,0,0] + M[...,0,1] * js) * jo - (M[...,1,0] + M[...,1,1] * js )
        
        r = numerator_r / denominator
        R = torch.abs(r) ** 2

        t_numerator = 2 * jo
        t = t_numerator / denominator
        T = (js / jo) * torch.abs(t) ** 2

        return R, T
    

    def Reflectance_from_layers(self, layers, theta_0=0, mode='TE'):
        # This function gives the reflectance
        
        # Infer device from the first layer's material
        device = layers[0].material.refractive_index.device
        theta = torch.as_tensor(theta_0, dtype=torch.float64, device=device)

        M = None


        for i in range(1, len(layers)-1):
            n_prev = layers[i-1].material.refractive_index
            n_curr = layers[i].material.refractive_index

            # angle inside current layer
            theta = self.snells_law(n_prev, n_curr, theta)

            M_layer = self.transfer_matrix(layers[i], theta, mode)
            if M is None:
                M = M_layer
            else:
                M = torch.matmul(M, M_layer)

        n_last = layers[-2].material.refractive_index
        n_sub  = layers[-1].material.refractive_index
        theta_f = self.snells_law(n_last, n_sub, theta)  

        R, T = self.Reflectance([layers[0],layers[-1]], M, [theta_0, theta_f], mode)
        return R, T
    

class PhotonicTransferMatrixVectorized(PhotonicTransferMatrix):
    '''
    Vectorized version of PhotonicTransferMatrix for batch processing
    [B, layer, wavelengths]
    list layers -> layer[B, layer_num, wavelengths]
                -> thickness[B, layer_num]
                -> material: refractive_index[B, layer_num, wavelengths]
                -> theta [B] or scalar


    Returns:       Reflectance [B, wavelengths]

    ----
    n0 sin(theta0) = n1 sin(theta1)
    n1 sin(theta1) = n2 sin(theta2)

    n2 sin(theta2) = n0 sin(theta0)


    n2 sin(theta2) = n3 sin(theta3)
    n3 sin(theta3) = n4 sin(theta4)
    n4 sin(theta4) = n5 sin(theta5)
    '''
    def __init__(self, *args, **kwargs):
        super(PhotonicTransferMatrixVectorized, self).__init__(*args, **kwargs)
        self.name = 'PhotonicTransferMatrixVectorized'
        self.pi         = np.pi


    

    
     


    

    