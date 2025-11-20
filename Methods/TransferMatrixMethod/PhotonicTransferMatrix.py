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
    
    def snells_law(self, n1, n2, theta1):
        # This function applies Snell's law: n1 * sin(theta1) = n2 * sin(theta2)
        theta1 = torch.as_tensor(theta1, dtype=torch.float64)
        n1 = torch.as_tensor(n1, dtype=torch.float64)
        n2 = torch.as_tensor(n2, dtype=torch.float64)
        
        sin_theta2 = (n1 / n2) * torch.sin(theta1)
        theta2 = torch.asin(sin_theta2)
        
        return theta2
    
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
    

    def Reflectance(self, boundary_layers, transfer_matrix, theta, mode='TE'): 
    # This function gives the reflectance of the structure

        assert len(boundary_layers) == 2 , "boundary_layers must be a list of two layers" 
        assert mode in ["TE","TM"], "mode must be TE or TM"
        
        theta_0 = torch.as_tensor(theta[0], dtype=torch.float64)
        theta_f = torch.as_tensor(theta[-1], dtype=torch.float64)

        permittivity_free_space = torch.as_tensor(8.854187817e-12)
        permeability_free_space = torch.as_tensor(1.2566370614e-6)
        jo = torch.sqrt(permittivity_free_space / permeability_free_space) * boundary_layers[0].material.refractive_index * torch.cos(theta_0)
        js = torch.sqrt(permittivity_free_space / permeability_free_space) * boundary_layers[1].material.refractive_index * torch.cos(theta_f)


        M = transfer_matrix

        denominator = (M[...,0,0] + M[...,0,1] * js) * jo + (M[...,1,0] + M[...,1,1] * js )

        numerator_r = (M[...,0,0] + M[...,0,1] * js) * jo - (M[...,1,0] + M[...,1,1] * js )
        
        r = numerator_r / denominator
        R = torch.abs(r)**2

        t_numerator = 2 * jo
        t = t_numerator / denominator
        T = (js / jo) * torch.abs(t) ** 2

        return R, T
    

    def Reflectance_from_layers(self, layers, theta_0=0, mode='TE'):
        # This function gives the reflectance

        theta_layer = self.snells_law(layers[0].material.refractive_index, layers[1].material.refractive_index, theta_0)
        M = self.transfer_matrix(layers[1],theta_layer, mode)

        for layer in layers[2:-1]:
            theta_layer = self.snells_law(layer.material.refractive_index, layers[layers.index(layer)+1].material.refractive_index, theta_layer)
            M = torch.matmul(M,self.transfer_matrix(layer,theta_layer,mode))
        
        theta_f = self.snells_law(layers[-2].material.refractive_index, layers[-1].material.refractive_index, theta_layer)

        R, T = self.Reflectance([layers[0],layers[-1]],M,[theta_0, theta_f],mode)
        return R, T
    



    

    
     


    

    