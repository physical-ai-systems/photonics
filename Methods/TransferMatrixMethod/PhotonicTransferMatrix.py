import torch
import numpy as np
class PhotonicTransferMatrix:
    def __init__(self,
                *args, **kwargs):
                 
        self.name       = 'PhotonicTransferMatrix'
        self.pi         = np.pi

    def p_value(self,material,mode, theta):
        # This function gives p value
        if   mode == "TE":
            p = torch.cos(theta) * material.permittivity/material.permeability
        elif mode == "TM":
            p = torch.cos(theta) * material.permeability/material.permittivity
        else:
            raise ValueError("mode must be TE or TM")
        return p
    
    def k_wavevector(self, material, theta):
        # This function gives k value
        k = material.wavelength.k * material.reference_index * torch.cos(theta)
        return k
    
    def transfer_matrix(self,layer,theta=0,mode='TE'):
        # This function gives the transfer matrix
        pi_2 = self.pi / 2
        k = self.k_wavevector(layer.material)
        p = self.p_value(layer.material,mode)

        
        M = torch.zeros((*k.shape,2,2), dtype = torch.complex128)
                
        M[...,0,0] = pi_2 * k_rho[...,0] * ( y_dash[...,0] * j[...,1] - j_dash[...,0] * y[...,1]) 
        M[...,0,1] = (  1j / p ) * pi_2 * k_rho[...,0] * ( j[...,0] * y[...,1] - y[...,0] * j[...,1]) 
        M[...,1,0] = ( -1j * p ) * pi_2 * k_rho[...,0] * ( y_dash[...,0] * j_dash[...,1] - j_dash[...,0] * y_dash[...,1]) 
        M[...,1,1] = pi_2 * k_rho[...,0] * ( j[...,0] * y_dash[...,1] - y[...,0] * j_dash[...,1]) 
      
        return M
    

    def Reflectance(self, boundary_layers, transfer_matrix, theta=0, mode='TE'): 
    # This function gives the reflectance of the structure

        assert len(boundary_layers) == 2 , "boundary_layers must be a list of two layers" 
        assert mode in ["TE","TM"], "mode must be TE or TM"

        rho      = torch.zeros(len(boundary_layers),2) # rho_beginning, rho_end
        k        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        p        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        c_factor = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,*rho.shape, dtype = torch.complex128)
        
        M = transfer_matrix

        for counter in range(len(boundary_layers)):
            layer = boundary_layers[counter]
            k[...,counter]      = self.k_wavevector(layer.material)
            p[...,counter]      = self.p_value(layer.material,mode)
            rho[counter,...]    = torch.as_tensor(layer.coordinates)
            c_factor[...,counter,0] = self.C_factor(1,theta,k[...,counter]*(rho[counter,1] if counter == 0 else rho[counter,0]))
            c_factor[...,counter,1] = self.C_factor(2,theta,k[...,counter]*(rho[counter,1] if counter == 0 else rho[counter,0]))

        numerator   = ( M[...,1,0] + 1j * p[...,0] *  c_factor[...,0,1] * M[...,0,0] ) - ( 1j * p[...,1] * c_factor[...,1,1] * ( M[...,1,1] + 1j * p[...,0] * c_factor[...,0,1] * M[...,0,1] ) ) 
        denominator = ( -1j * p[...,0] * c_factor[...,0,0] * M[...,0,0] - M[...,1,0] ) - ( 1j * p[...,1] * c_factor[...,1,1] * ( -1j * p[...,0] * c_factor[...,0,0] * M[...,0,1] - M[...,1,1] ) )

        r = numerator / denominator
        R = torch.abs(r)**2
        return R
    

    def Reflectance_from_layers(self, layers, theta=0, mode='TE'):
        # This function gives the reflectance
        M = self.transfer_matrix(layers[1],theta,mode)
        for layer in layers[2:-1]:
            M = torch.matmul(M,self.transfer_matrix(layer,theta,mode))
        R = self.Reflectance([layers[0],layers[-1]],M,theta,mode)
        return R
    



    

    
     


    

    