import torch
import copy
import torch.nn as nn
class Structure(nn.Module):
    def __init__(self,
                 layers,
                 name=None,
                 mode="TE",
                 layers_parameters = {'method':'repeat', 
                                      'layers_repeat':[1,10,1],
                                      },
                 *args, **kwargs):
        super().__init__()
        self.name = name
        self.mode = mode
        self.layers      = self.structure_init(layers, **layers_parameters, **kwargs)
        self.coordinates = None
    

    def structure_init(self, layers, method='repeat', layers_repeat=[1,10,1], **kwargs):
        if method == 'repeat':
            self.layers = self.repeat_layers(layers, layers_repeat)     
        else:
            raise NotImplementedError("Method {} is not implemented".format(method))           
        
        self.get_coordinates()

        return self.layers
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def get_coordinates(self):
        coordinate = 0
        for layer in self.layers:
            if layer.thickness is not None:
                layer.coordinates = copy.deepcopy([coordinate, coordinate + layer.thickness]) 
                coordinate = coordinate + layer.thickness
            else:
                raise ValueError("Layer thickness is not defined")

        return self.layers
    
    def repeat_layers(self, layers_base, repeat_layers):
        layers = []
        for i in range(len(repeat_layers)):
            if isinstance(repeat_layers[i], list):
                layers.extend(self.repeat_layers(layers_base[i], repeat_layers[i]))
            elif isinstance(repeat_layers[i], int):
                if not isinstance(layers_base[i], list):
                    layers.extend([copy.copy(item)for item in ([layers_base[i]] * repeat_layers[i])])
                else:
                    layers.extend([copy.copy(item)for item in (layers_base[i] * repeat_layers[i])])
            else:
                raise NotImplementedError("Repeat layers {} is not implemented".format(repeat_layers))
        return layers
    
    def __repr__(self):
        return "Structure name: {}, Layers: {}".format(self.name, self.layers)
    
    def __str__(self):
        return "Structure name: {}, Layers: {}".format(self.name, self.layers)
    
    def __len__(self):
        return len(self.layers)
    
    def calculate(self, method, *args, **kwargs):
        if (method.name == 'TransferMatrixMethod' or method.name =='AnnularPhotonicTransferMatrix') and (not self.layers[1].store_transfer_matrix):
            M = torch.eye(2,2,dtype=torch.complex128).unsqueeze(0).repeat(self.layers[1].material.refractive_index.shape[0],1,1)
            for layer in self.layers[1:-1]: # skip first and last layer
                M = M @ method.transfer_matrix(layer)
            return M
        elif (method.name == 'TransferMatrixMethod' or method.name =='AnnularPhotonicTransferMatrix') and (self.layers[1].store_transfer_matrix):
            M = torch.eye(*self.layer[1].material.refractive_index.shape[0],2,2)
            for layer in self.layers[1:-1]: # skip first and last layer
                M = M @ layer.transfer_matrix
            return M
        elif method == 'FDTD':
            raise NotImplementedError("Method {} is not implemented".format(method))
        else:
            raise NotImplementedError("Method {} is not implemented".format(method))

    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self
    @property
    def device(self):
        return self.layers[0].device