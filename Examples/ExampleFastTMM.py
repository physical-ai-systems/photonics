import sys 
import os
import torch

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))

from Methods.TransferMatrixMethod.Wavelength             import WaveLength
from Methods.TransferMatrixMethod.Structure              import Structure
from Methods.TransferMatrixMethod.Layer                  import Layer
from Materials.Materials                                 import Material 
from Methods.TransferMatrixMethod.PhotonicTransferMatrixFast import PhotonicTransferMatrixFast
from Methods.PhysicalQuantity                            import PhysicalQuantity
from Visualization.Visualization                         import plot

def stack_layers(layers_list, wavelength):
    # Extract refractive indices and thicknesses
    ris = []
    thicknesses = []
    
    for layer in layers_list:
        # Ensure refractive index is a tensor with correct shape
        ri = layer.material.refractive_index
        ris.append(ri)
        
        t = layer.thickness
        
        # Ensure t is [batch, 1]
        batch_size = ri.shape[0]
        
        if t.ndim == 0:
            t = t.view(1)
            
        if t.shape[0] == 1 and batch_size > 1:
            t = t.expand(batch_size)
            
        if t.ndim == 1:
            t = t.unsqueeze(1) # [batch, 1]
            
        thicknesses.append(t)

    # Stack
    # ris: list of [batch, wl]
    # stacked_ri: [batch, N, wl]
    stacked_ri = torch.stack(ris, dim=1)
    
    # thicknesses: list of [batch, 1]
    # stacked_t: [batch, N, 1]
    stacked_t = torch.stack(thicknesses, dim=1)
    
    # Create super material
    # We need a name, let's say "StackedMaterial"
    # We pass the stacked RI.
    super_material = Material(wavelength, name="StackedMaterial", refractive_index=stacked_ri)
    
    # Create super layer
    super_layer = Layer(super_material, thickness=stacked_t)
    
    return super_layer

def material_sensor(wavelength, method,
                            layer_1_thickness,
                            layer_2_thickness,
                            batch_size,
                            N,
                            layer_defect_thickness,
                            **kwargs):
    
    # broadcast the values
    wavelength.broadcast([batch_size, *wavelength.values.shape])
    
    SiO2               = Material(wavelength, name="SiO2", refractive_index=1.4618)
    Air                = Material(wavelength, name="Air",  refractive_index=1)

    A                  = Material(wavelength, name='A', refractive_index=3.3)
    MgF2               = Material(wavelength, name="MgF2", refractive_index=1.3855)
    glass              = Material(wavelength, name='glass', refractive_index=1.52)

    # Define the layers
    layer0 = Layer(Air)
    layer1 = Layer(A,  thickness=layer_1_thickness)
    layer2 = Layer(MgF2, thickness=layer_2_thickness)
    layerd = Layer(SiO2, thickness=layer_defect_thickness)
    layerf = Layer(glass)

    # Construct the list of internal layers
    # [layer1, layer2] * N, layerd, [layer1, layer2] * N
    internal_layers_list = []
    for _ in range(N):
        internal_layers_list.append(layer1)
        internal_layers_list.append(layer2)
    
    internal_layers_list.append(layerd)
    
    for _ in range(N):
        internal_layers_list.append(layer1)
        internal_layers_list.append(layer2)
        
    # Stack internal layers
    stacked_internal_layer = stack_layers(internal_layers_list, wavelength)
    
    # Stack boundary layers [layer0, layerf]
    boundary_layers_list = [layer0, layerf]
    stacked_boundary_layer = stack_layers(boundary_layers_list, wavelength)
    
    print(f"Total number of internal layers: {len(internal_layers_list)}")
    
    # Call the fast method
    R, T = method.Reflectance_from_layers(stacked_internal_layer, stacked_boundary_layer, theta_0=torch.tensor(0.0), mode= 'TE')
    
    outputs = {}
    outputs['Reflectance']  = PhysicalQuantity(values=R, units='a.u.', name='Reflectance')
    outputs['Transmission'] = PhysicalQuantity(values=T, units='a.u.', name='Transmission')

    return outputs

def get_parameters():
    method = PhotonicTransferMatrixFast()
    batch_size = 1
    wavelength = WaveLength(ranges=[150,600], steps=0.005, units="m", unit_prefix="n")
    layer_1_thickness = 22.73   * 1e-9 # m 
    layer_2_thickness = 54.13   * 1e-9 # m
    layer_defect_thickness = 51.31 * 1e-9
    parameters = {
        'wavelength':wavelength, 
        'method':method,
        'layer_1_thickness':layer_1_thickness, 'layer_2_thickness':layer_2_thickness,
        'layer_defect_thickness': layer_defect_thickness,
        'batch_size':batch_size,
        'N': 5
        }
    
    return parameters

if __name__ == "__main__":

    parameters = get_parameters()
    outputs = material_sensor(**parameters)
    
    wavelength_nm = parameters['wavelength'].values.squeeze() * 1e9
    
    plot(
        x=wavelength_nm,
        y=outputs['Transmission'].values.squeeze(),
        names=['Transmission'],
        x_label='Wavelength (nm)',
        y_label='Transmission (a.u.)',
        show=True,
        width=1600,
        height=800
    )
