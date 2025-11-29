import torch
import torch.nn as nn
class Layer(nn.Module):
    def __init__(self, material, method=None, thickness=None, store_transfer_matrix=False, *args, **kwargs):
        super().__init__()
        self.material               = material
        self.method                 = method
        self.thickness = None if thickness is None else thickness if isinstance(thickness, torch.Tensor) else torch.as_tensor(thickness)

        self.thickness_round_digits  = 0 # round thickness to n digits, please note that the length of the wavelength scale is added to this number
        self.thickness_round_digits += int((-1*torch.log10(torch.as_tensor(self.material.wavelength.scale)).item())) # add the length of the wavelength scale to the round digits
        self.coordinates             = None
        self.transfer_matrix         = None
        self.store_transfer_matrix   = store_transfer_matrix
        self.get_thickness()

        if store_transfer_matrix and method is not None:
            self.transfer_matrix = method.TransfertMatrix(self)   

    def get_thickness(self):
        self.suggest_thickness = self.material.wavelength.center / (4 * self.material.refractive_index_avarage_real_part)
        self.suggest_thickness = torch.round(self.suggest_thickness, decimals= self.thickness_round_digits) # round to n digits
        if self.thickness is None:
            self.thickness = self.suggest_thickness
        if len(self.thickness.shape) == 0:
            self.thickness = self.thickness.unsqueeze(0)

    def __repr__(self):
        return "Layer: {} with thickness: {}".format(self.material.name, self.thickness)

    def to(self, device):
        self.material.to(device)
        self.thickness = self.thickness.to(device)
        return self
    
    @property
    def device(self):
        return self.material.device
    

    
class MultiLayer(nn.Module):
    def __init__(self, material, thickness=None, *args, **kwargs):
        super().__init__()
        self.material   = material
        self.thickness  = thickness 

    def __repr__(self):
        return "Layer: {} with thickness: {}".format(self.material.name, self.thickness)

    def to(self, device):
        self.material.to(device)
        self.thickness = self.thickness.to(device)
        return self
    
    @property
    def device(self):
        return self.material.device
    

     
