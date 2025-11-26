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
        elif method == 'multi_layer':
            self.layers = layers
        else:
            raise NotImplementedError("Method {} is not implemented".format(method))           
        
        self.get_coordinates()

        return self.layers
    
    def add_layer(self, layer):
        self.layers.append(layer)
        
    def get_coordinates(self):
        coordinate = 0
        for i, layer in enumerate(self.layers):
            if layer.thickness is not None:
                # Debug print for device mismatch
                # th_dev = layer.thickness.device if isinstance(layer.thickness, torch.Tensor) else "cpu"
                # coord_dev = coordinate.device if isinstance(coordinate, torch.Tensor) else "int"
                # print(f"Layer {i} ({layer.material.name}): Coord dev={coord_dev}, Thick dev={th_dev}")
                
                # Ensure coordinate is on the same device as layer.thickness if it's 0 (int)
                if isinstance(coordinate, int) and coordinate == 0 and isinstance(layer.thickness, torch.Tensor):
                    coordinate = torch.tensor(0.0, device=layer.thickness.device, dtype=layer.thickness.dtype)

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
    
    def to(self, device):
        for layer in self.layers:
            layer.to(device)
        return self
    
    @property
    def device(self):
        return self.layers[0].device