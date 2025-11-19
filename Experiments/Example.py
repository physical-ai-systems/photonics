import sys 
import os

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import torch
from Methods.TransferMatrixMethod.Wavelength import WaveLength
from Methods.TransferMatrixMethod.Structure  import Structure
from Methods.TransferMatrixMethod.Layer      import Layer
from Materials.Materials                     import Material 
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Study.Study                             import SaveData, LoadData
from Methods.PhysicalQuantity                import PhysicalQuantity
import traceback

def material_sensor(wavelength, method,
                            layer_1_thickness,
                            layer_2_thickness,
                            batch_size,
                            **kwargs):
    
    # get the variable material

    # broadcast the values
    wavelength.broadcast([batch_size, *wavelength.values.shape])
    
    Si                 = Material(wavelength, name="Si",   refractive_index=3.7)
    SiO2               = Material(wavelength, name="SiO2", refractive_index=1.45)
    Air                = Material(wavelength, name="Air",  refractive_index=1)

    # Define the layers
    layer0 = Layer(Air)
    layer1 = Layer(SiO2,  thickness=layer_1_thickness)
    layer2 = Layer(Si, thickness=layer_2_thickness)
    layerf = Layer(SiO2, thickness=1000e-9)

    # Define the structure
    structure = Structure([layer0, layer1, layer2, layerf],
                                        layers_parameters = {'method':'repeat', 
                                                                'layers_repeat':[1,1,1,1],
                                                                })                                        # transfer the structure to the device   structure.to(method.device)
   

    values = method.Reflectance_from_layers(structure.layers, m=0, mode= 'TE')
    # get the reflectance
    outputs = {}
    outputs['Reflectance']  = PhysicalQuantity(values = values, units='a.u.', name='Reflectance')

    return outputs

def get_parameters():
    method = PhotonicTransferMatrix()
    batch_size = 1
    wavelength = WaveLength(ranges=[420,750],steps=0.005, units="m", unit_prefix="n")
    layer_1_thickness = 50   * 1e-9 # m 
    layer_2_thickness = 70   * 1e-9 # m

    parameters = {
        'wavelength':wavelength, 
        'method':method,
        'layer_1_thickness':layer_1_thickness, 'layer_2_thickness':layer_2_thickness,
        'batch_size':batch_size,
        }
    
    return parameters



if __name__ == "__main__":

    parameters = get_parameters()
    outputs = material_sensor(**parameters)
