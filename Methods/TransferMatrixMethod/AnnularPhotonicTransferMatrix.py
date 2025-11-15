import torch
import numpy as np
import torch.nn as nn

# try: 
#     import cupy as backend
#     device = 'cuda'
# except:
#     import scipy as backend
#     device = 'cpu'

# current cupy version does not support all the special functions required for the implementation

import scipy as backend
device = 'cpu'

class AnnularPhotonicTransferMatrix(nn.Module):
    def __init__(self,
                *args, **kwargs):
        super().__init__()
                 
        self.name       = 'AnnularPhotonicTransferMatrix'
        self.pi         = np.pi
        self.device     = device
    
    def from_numpy(self, x):
        if self.device == 'cuda':
            return torch.from_dlpack(x)
        else:
            return torch.as_tensor(x)
    
    def to_numpy(self, x):
        if self.device == 'cuda':
            return backend.from_dlpack(x)
        else:
            return x.numpy(force = True)

    def bessel_j(self, m, z):
        return self.from_numpy(backend.special.jv(m, self.to_numpy(z)))
    
    def bessel_y(self, m, z):
        return self.from_numpy(backend.special.yv(m, self.to_numpy(z)))

    def bessel_h(self,n, m, z):
        if n == 1:
            return self.from_numpy(backend.special.hankel1(m, self.to_numpy(z)))
        elif n == 2:
            return self.from_numpy(backend.special.hankel2(m, self.to_numpy(z)))
        else:
            raise ValueError("n must be 1 or 2")
        
    def bessel_y_dash(self, m,z):
        '''
        This function computes the derivative of the Bessel function of the second kind.
        args:
            input:
                m: order of the Bessel function
                z: input value

            output:
                J: derivative of the Bessel function of the second kind

        '''
        assert type(m) is int and m >= 0, 'order must be a positive integer for current implementation; Please extend the implementation for any real number'
        if m == 0:
            Y_dash = -self.bessel_y(1,z)
        else:
            Y_dash = 0.5 * (self.bessel_y(m-1,z) - self.bessel_y(m+1,z))
        return Y_dash 
    
    def bessel_j_dash(self, m,z):
        '''
        This function computes the derivative of the Bessel function of the first kind.
        args:
            input:
                m: order of the Bessel function
                z: input value

            output:
                J: derivative of the Bessel function of the first kind

        '''
        assert type(m) is int and m >= 0, 'm must be a positive integer for current implementation; Please extend the implementation for any real number'
        if m == 0:
            J_dash = -self.bessel_j(1,z)
        else:
            J_dash = 0.5 * (self.bessel_j(m-1,z) - self.bessel_j(m+1,z))
        return J_dash
    
    # def bessel_h(self, n, m, z):
    #     '''
    #     This function computes the Hankel function of the kind n, where n = 1,2.
    #     args:
    #         input:
    #             n: kind of the Hankel function
    #             m: order of the Hankel function
    #             z: input value

    #         output:
    #             J: Hankel function of the kind n 
    #     '''
    #     if n == 1:
    #         return self.bessel_j(m,z) + 1j * self.bessel_y(m,z)
    #     elif n == 2:
    #         return self.bessel_j(m,z) - 1j * self.bessel_y(m,z)
    #     else:
    #         raise Exception('Please use a valid kind of the Hankel function')
        
    def bessel_h_dash(self, n, m, z):
        '''
        This function computes the derivative of the Hankel function of the kind n, where n =1,2.
        args:
            input:
                n: kind of the Hankel function
                m: order of the Hankel function
                z: input value

            output:
                J: derivative of the Hankel function of the kind n 
        '''
        if n == 1:
            return self.bessel_j_dash(m,z) + 1j * self.bessel_y_dash(m,z)
        elif n == 2:
            return self.bessel_j_dash(m,z) - 1j * self.bessel_y_dash(m,z)
        else:
            raise Exception('Please use a valid kind of the Hankel function')
        

    def C_factor(self,n, m, z):
        # This function gives C_factor
        if n == 1 or n == 2:
            H = self.bessel_h(n, m, z)
            H_dash = self.bessel_h_dash(n, m, z)
            C = H_dash / H
        else:
            raise ValueError('The kind of the Hankel function must be 1 or 2')
        return C
    
    def p_value(self,material,mode):
        # This function gives p value
        if   mode == "TE":
            p = torch.sqrt(material.permittivity/material.permeability)
        elif mode == "TM":
            p = torch.sqrt(material.permeability/material.permittivity)   
        else:
            raise ValueError("mode must be TE or TM")
        return p
    
    def k_wavevector(self,material):
        # This function gives k value
        k = material.wavelength.k * material.wavelength.speed_of_light * torch.sqrt(material.permittivity * material.permeability)    
        return k
    
    def transfer_matrix(self,layer,m=0,mode='TE'):
        # This function gives the transfer matrix
        pi_2 = self.pi / 2
        k = self.k_wavevector(layer.material)
        p = self.p_value(layer.material,mode)
        k_rho = torch.zeros([*k.shape, 2])
        k_rho[...,0] = k * torch.as_tensor(layer.coordinates[0])
        k_rho[...,1] = k * torch.as_tensor(layer.coordinates[1])
        j      = self.bessel_j(m,k_rho)
        j_dash = self.bessel_j_dash(m,k_rho)
        y      = self.bessel_y(m,k_rho)
        y_dash = self.bessel_y_dash(m,k_rho)
        
        M = torch.zeros((*k.shape,2,2), dtype = torch.complex128)
            
        M[...,0,0] = pi_2 * k_rho[...,0] * ( y_dash[...,0] * j[...,1] - j_dash[...,0] * y[...,1]) 
        M[...,0,1] = (  1j / p ) * pi_2 * k_rho[...,0] * ( j[...,0] * y[...,1] - y[...,0] * j[...,1]) 
        M[...,1,0] = ( -1j * p ) * pi_2 * k_rho[...,0] * ( y_dash[...,0] * j_dash[...,1] - j_dash[...,0] * y_dash[...,1]) 
        M[...,1,1] = pi_2 * k_rho[...,0] * ( j[...,0] * y_dash[...,1] - y[...,0] * j_dash[...,1]) 
      
        return M
    

    def Reflectance(self, boundary_layers, transfer_matrix, m=0, mode='TE'): 
    # This function gives the reflectance of the structure

        assert len(boundary_layers) == 2 , "boundary_layers must be a list of two layers" 
        assert mode in ["TE","TM"], "mode must be TE or TM"

        rho      = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers),2) # rho_beginning, rho_end
        k        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        p        = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers))
        c_factor = torch.zeros(*boundary_layers[0].material.wavelength.values.shape,len(boundary_layers),2, dtype = torch.complex128)
        
        M = transfer_matrix

        for counter in range(len(boundary_layers)):
            layer = boundary_layers[counter]
            k[...,counter]      = self.k_wavevector(layer.material)
            p[...,counter]      = self.p_value(layer.material,mode)
            rho[...,counter,0]    = torch.as_tensor(layer.coordinates[0])
            rho[...,counter,1]    = torch.as_tensor(layer.coordinates[1])
            c_factor[...,counter,0] = self.C_factor(1,m,k[...,counter]*(rho[...,counter,1] if counter == 0 else rho[...,counter,0]))
            c_factor[...,counter,1] = self.C_factor(2,m,k[...,counter]*(rho[...,counter,1] if counter == 0 else rho[...,counter,0]))

        numerator   = ( M[...,1,0] + 1j * p[...,0] *  c_factor[...,0,1] * M[...,0,0] ) - ( 1j * p[...,1] * c_factor[...,1,1] * ( M[...,1,1] + 1j * p[...,0] * c_factor[...,0,1] * M[...,0,1] ) ) 
        denominator = ( -1j * p[...,0] * c_factor[...,0,0] * M[...,0,0] - M[...,1,0] ) - ( 1j * p[...,1] * c_factor[...,1,1] * ( -1j * p[...,0] * c_factor[...,0,0] * M[...,0,1] - M[...,1,1] ) )

        r = numerator / denominator
        R = torch.abs(r)**2
        return R
    

    def Reflectance_from_layers(self, layers, m=0, mode='TE'):
        # This function gives the reflectance
        M = self.transfer_matrix(layers[1],m,mode)
        for layer in layers[2:-1]:
            M = torch.matmul(M,self.transfer_matrix(layer,m,mode))
        R = self.Reflectance([layers[0],layers[-1]],M,m,mode)
        return R
    



    

    
     


    

    