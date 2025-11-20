import torch
import numpy as np
class PhotonicTransferMatrix:
    def __init__(self,
                *args, **kwargs):
                 
        self.name       = 'PhotonicTransferMatrix'
        self.pi         = np.pi

    def p_value(self,material,mode, theta):
        # This function gives p value
        theta = torch.as_tensor(theta, dtype=torch.float64)
        if   mode == "TE":
            p = torch.cos(theta) * material.refractive_index
        elif mode == "TM":
            p = torch.cos(theta) / material.refractive_index
        else:
            raise ValueError("mode must be TE or TM")
        return p
    
    def k_wavevector(self, material, theta):
        # This function gives k value
        k = material.wavelength.k * material.refractive_index
        return k
    
    def transfer_matrix(self,layer,theta=0,mode='TE'):
        # This function gives the transfer matrix
        theta = torch.as_tensor(theta, dtype=torch.float64)
        k = self.k_wavevector(layer.material, theta)
        p = self.p_value(layer.material,mode, theta)

        
        M = torch.zeros((*k.shape,2,2), dtype = torch.complex128)
        delta = layer.thickness * layer.material.refractive_index * torch.cos(theta)
        
        M[...,0,0] = torch.cos(k * delta)
        M[...,0,1] = -(  1j / p ) * torch.sin(k * delta)
        M[...,1,0] = ( -1j * p ) * torch.sin(k * delta) 
        M[...,1,1] = torch.cos(k * delta)
      
        return M
    

    def Reflectance(self, boundary_layers, transfer_matrix, theta=0, mode='TE'): 
    # This function gives the reflectance of the structure

        assert len(boundary_layers) == 2 , "boundary_layers must be a list of two layers" 
        assert mode in ["TE","TM"], "mode must be TE or TM"
        
        theta = torch.as_tensor(theta, dtype=torch.float64)
        rho      = torch.zeros(len(boundary_layers),2) # rho_beginning, rho_end
        k        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        p        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        

        permittivity_free_space = torch.as_tensor(8.854187817e-12)
        permeability_free_space = torch.as_tensor(1.2566370614e-6)
        jo = torch.sqrt(permittivity_free_space / permeability_free_space) * boundary_layers[0].material.refractive_index * torch.cos(theta)
        js = torch.sqrt(permittivity_free_space / permeability_free_space) * boundary_layers[1].material.refractive_index * torch.cos(theta)


        M = transfer_matrix

        numerator = (M[...,0,0] + M[...,0,1] * js) * jo - (M[...,1,0] + M[...,1,1] * js )
        denominator = (M[...,0,0] + M[...,0,1] * js) * jo + (M[...,1,0] + M[...,1,1] * js )
        
        r = numerator / denominator
        R = torch.abs(r)**2

        t_numerator = 2 * jo
        t = t_numerator / denominator
        T = (js / jo) * torch.abs(t) ** 2

        return R, T
    

    def Reflectance_from_layers(self, layers, theta=0, mode='TE'):
        # This function gives the reflectance
        M = self.transfer_matrix(layers[1],theta,mode)
        for layer in layers[2:-1]:
            M = torch.matmul(M,self.transfer_matrix(layer,theta,mode))
        R, T = self.Reflectance([layers[0],layers[-1]],M,theta,mode)
        return R, T
    



    

    
     


    

    