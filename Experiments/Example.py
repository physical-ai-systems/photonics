import sys 
import os
import matplotlib.pyplot as plt

try:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
except:
    sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
import torch
from Methods.TransferMatrixMethod.Wavelength import WaveLength
from Methods.TransferMatrixMethod.Structure  import Structure
from Methods.TransferMatrixMethod.Layer      import Layer
from Materials.Materials                     import Material 
from Materials.Materials                     import CompositeMaterial
from Methods.TransferMatrixMethod.AnnularPhotonicTransferMatrix import AnnularPhotonicTransferMatrix
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Materials.Materials_database.Materials.MaterialsFunctions.PVA import refractive_index_PVA
from Study.Study                             import SaveData, LoadData
from Methods.PhysicalQuantity                import PhysicalQuantity
import traceback

def material_sensor(wavelength, method,
                            N, rho0,
                            layer_1_thickness, layer_2_thickness, defect_thickness,
                            porosity_1,
                            variableProperty,
                            batch_size,
                            **kwargs):
    
    
    # broadcast the values
    wavelength.broadcast([batch_size,len(variableProperty.values), *wavelength.values.shape])
    
    lossless           = Material(wavelength, name="lossless",   refractive_index=3.3)
    Si                 = Material(wavelength, name="Si",   refractive_index=1.3855)
    SiO2               = Material(wavelength, name="SiO2", refractive_index=1.4618)
    #Air                = Material(wavelength, name="Air",  refractive_index=1)
    #material_porose_1 = CompositeMaterial.get_material('PorousMedium', [Si,variableMaterial], porosity=porosity_1)

    # Define the layers
    #layer0 = Layer(SiO2,thickness=rho0)
    #layerd = Layer(variableMaterial,  thickness=defect_thickness)
    #layer1 = Layer(material_porose_1, thickness=layer_1_thickness)
    #layer2 = Layer(variableMaterial,  thickness=layer_2_thickness)
    #layerf = Layer(SiO2, thickness=1000e-9)
    layerA = Layer(lossless, thickness=22.73e-9)
    layerB = Layer(Si, thickness=54.13e-9)
    layerD = Layer(SiO2, thickness=51.31e-9)

    # Define the structure
    structure = Structure([[layerA, layerB], layerD ,[layerA, layerB]],
                                        layers_parameters = {'method':'repeat', 
                                                                'layers_repeat':[int(N),1,int(N)],
                                                                })                                        # transfer the structure to the device   structure.to(method.device)
   

    values = method.Reflectance_from_layers(structure.layers, m=0, mode= 'TE')
    # get the reflectance
    outputs = {}
    outputs['Reflectance']  = PhysicalQuantity(values = values, units='a.u.', name='Reflectance')

    return outputs

def get_parameters():
    # Define the experiment discription
    try :
        experiment_path = os.path.dirname(__file__)
        experiment_name = os.path.basename(__file__).split('.')[0]
    except Exception as e:
        experiment_path = os.path.join(os.getcwd())
        experiment_name = 'Example'

    experiment_path, experiment_name = SaveData.create_experiment_description(experiment_path=experiment_path, experiment_name=experiment_name)
    # Define the method
    method = PhotonicTransferMatrix()
    batch_size = 1
    wavelength = WaveLength(ranges=[300,300],steps=0.005, units="m", unit_prefix="n")

    N                 =  10
    rho0              =  500 * 1e-9 # m
    porosity_1        =  torch.tensor(0.7)
    layer_1_thickness = 22.73e-9 # m 
    layer_2_thickness = 54.13e-9 # m
    defect_thickness  = 51.31e-9 # m
    variableProperty   = PhysicalQuantity(values=torch.tensor([0,70]), units='Gy', name='$\gamma$  dose') # Gamma ray dose in Gy

    parameters = {
        'wavelength':wavelength, 'experiment_path':experiment_path, 'method':method,
        'N':N, 'rho0':rho0,
        'layer_1_thickness':layer_1_thickness, 'layer_2_thickness':layer_2_thickness, 'defect_thickness':defect_thickness,
        'porosity_1':porosity_1,
        'variableProperty':variableProperty, 
        'batch_size':batch_size,
        }
    
    parameters_analysis = {
        'wavelength':wavelength, 'experiment_path':experiment_path,
        # 'PBG_range':[420,578],
        'PBG_range':None,
        'variableProperty':variableProperty,  
        'working_dim':-1, 'referance_index':0,
        'backend':'scipy', 'strcuture_type':'periodic_with_defect', 'number_of_defect_peaks':1,
        }
    return parameters, parameters_analysis, experiment_path, experiment_name



if __name__ == "__main__":

    parameters, _, experiment_path, experiment_name = get_parameters()
    outputs = material_sensor(**parameters)
