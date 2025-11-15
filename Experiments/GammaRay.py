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
from Materials.Materials_database.Materials.MaterialsFunctions.PVA import refractive_index_PVA
from Study.OutputAnalysis                    import Analysis
from Study.Study                             import StudySwipe
from Study.Study                             import SaveData, LoadData
from Methods.PhysicalQuantity                import PhysicalQuantity
from Visualization.Visualization             import visualization
from Visualization.Visualization             import plot
from Utils.Utils                             import batched_values
from Utils.Export                            import ExportUtils
import traceback

def get_variable_material(wavelength, variableProperty, batch_size, **kwargs):
    refractive_index_variable_material = refractive_index_PVA(wavelength.values*1e9, variableProperty.values)
    refractive_index_variable_material = refractive_index_variable_material.unsqueeze(0).repeat(batch_size,1,1)
    variableMaterial                   = Material(wavelength, name="PVA", refractive_index=refractive_index_variable_material)
    property_value                     = PhysicalQuantity(values=refractive_index_variable_material, units='RIU', name='Refractive index')
    return variableMaterial, property_value


def material_sensor(wavelength, method,
                            N, rho0,
                            layer_1_thickness, layer_2_thickness, defect_thickness,
                            porosity_1,
                            variableProperty,
                            batch_size,
                            **kwargs):
    
    # get the variable material
    variableMaterial, _ = get_variable_material(wavelength, variableProperty, batch_size)

    # broadcast the values
    wavelength.broadcast([batch_size,len(variableProperty.values), *wavelength.values.shape])
    
    Si                 = Material(wavelength, name="Si",   refractive_index=3.7)
    SiO2               = Material(wavelength, name="SiO2", refractive_index=1.45)
    Air                = Material(wavelength, name="Air",  refractive_index=1)
    material_porose_1 = CompositeMaterial.get_material('PorousMedium', [Si,variableMaterial], porosity=porosity_1)

    # Define the layers
    layer0 = Layer(SiO2,thickness=rho0)
    layerd = Layer(variableMaterial,  thickness=defect_thickness)
    layer1 = Layer(material_porose_1, thickness=layer_1_thickness)
    layer2 = Layer(variableMaterial,  thickness=layer_2_thickness)
    layerf = Layer(SiO2, thickness=1000e-9)
    # print('Layer 1 suggested thickness:', layer1.suggest_thickness, 'Layer 2 suggested thickness:', layer2.suggest_thickness, 'Defect layer suggested thickness:', layerd.suggest_thickness)

    # Define the structure
    structure = Structure([layer0, [layer1, layer2], layerd ,[layer1, layer2],layerf],
                                        layers_parameters = {'method':'repeat', 
                                                                'layers_repeat':[1,int(N),1,int(N),1],
                                                                })                                        # transfer the structure to the device   structure.to(method.device)
   
    # get the reflectance
    outputs = {}
    outputs['Reflectance']  = PhysicalQuantity(values = method.Reflectance_from_layers(structure.layers, m=0, mode= 'TE'), units='a.u.', name='Reflectance')

    return outputs

def analysis(wavelength,
            Results=None,           
            experiment_path=None, load_name=None, 
            PBG_range = None,
            variableProperty = None,
            working_dim = None,referance_index = None,
            batch_size = 1,
            backend='scipy',
            strcuture_type = 'periodic_with_defect', number_of_defect_peaks = 1,
            **kwargs):

    analysis = Analysis()


    if Results is not None:
        R = Results['Reflectance']
    elif experiment_path is not None:
        data = LoadData.get_reference_data(experiment_path, load_name,)
        keys = list(data.keys())
        for key in keys:
            if 'Reflectance' in key:
                R = PhysicalQuantity(values=data[key], units='a.u.', name='Reflectance')
    else:
        raise ValueError('Results and save_dir cannot be None at the same time')



    _, property_value = get_variable_material(wavelength, variableProperty, batch_size)



    wavelength.set_values_in_unit()


    try:
        output = analysis.get_output(wavelength=wavelength, R=R, PBG_range=PBG_range, backend=backend, strcuture_type=strcuture_type, number_of_defect_peaks=number_of_defect_peaks, property_value=property_value, working_dim=working_dim, referance_index=referance_index)
        return output
    except Exception as e:
        print('Error in analysis:', e)
        traceback.print_exc()
        return None

def get_parameters():
    # Define the experiment discription
    try :
        experiment_path = os.path.dirname(__file__)
        experiment_name = os.path.basename(__file__).split('.')[0]
    except Exception as e:
        experiment_path = os.path.join(os.getcwd())
        experiment_name = 'GammaRay'

    experiment_path, experiment_name = SaveData.create_experiment_description(experiment_path=experiment_path, experiment_name=experiment_name)
    # Define the method
    method = AnnularPhotonicTransferMatrix()
    batch_size = 1
    wavelength = WaveLength(ranges=[420,750],steps=0.005, units="m", unit_prefix="n")
    # wavelength = WaveLength(ranges=[400,800],steps=5, units="m", unit_prefix="n")

    N                 =  20
    rho0              =  500 * 1e-9 # m
    porosity_1        =  torch.tensor(0.7)
    layer_1_thickness = 50   * 1e-9 # m 
    layer_2_thickness = 70   * 1e-9 # m
    defect_thickness  = 25  * 1e-9 # m
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

    parameters, parameters_analysis, experiment_path, experiment_name = get_parameters()
    # outputs = material_sensor(**parameters)

    # Sweep the parameters
    batch_size = 1

    physical_quantity={
        # 'defect_thickness' :PhysicalQuantity(batched_values(55-30,55+30,100,55), units='m', unit_prefix='n', name='Defect thickness'),
        # 'layer_1_thickness':PhysicalQuantity(ranges=[30,70], steps=0.1, units='m', unit_prefix='n', name='Layer a thickness'),
        # 'layer_2_thickness':PhysicalQuantity(ranges=[30,80], steps=0.1, units='m', unit_prefix='n', name='Layer b thickness'),
        'porosity_1'       :PhysicalQuantity(ranges=[0.2,0.7], steps= 0.01, units=None, unit_prefix=None, name='Layer a porosity'),
        # 'porosity_2'       :PhysicalQuantity(batched_values(0.25,0.35,50,0.3), units=None, unit_prefix=None, name='Layer b porosity'),
        'rho0'             :PhysicalQuantity(ranges=[50,700], steps=5, units='m', unit_prefix='n', name='Inner core radius'),
        # 'N'                :PhysicalQuantity(batched_values(5,15,10,10), units=None, unit_prefix=None, name='Number of periods'),
                    }
    StudySwipe.get_output(study_function=material_sensor,parameter_dict=parameters,physical_quantity=physical_quantity,batch_size=batch_size,
                                                                                                                        output_shape=[parameters['variableProperty'].values.shape[-1],
                                                                                                                        parameters['wavelength'].values.shape[-1]]) 

    StudySwipe.get_output(study_function=analysis,parameter_dict=parameters_analysis,physical_quantity=physical_quantity,batch_size=batch_size,
                                                                                                                        save_name_suffix='_analysis',
                                                                                                                        study_type='once',                                                                                                                    
                                                                                                                        output_shape=[parameters['variableProperty'].values.shape[-1],
                                                                                                                        parameters['wavelength'].values.shape[-1]],)


    
    # visualization.visualize(experiment_path, physical_quantity)
    # ExportUtils.export_images_to_word( experiment_path, physical_quantity, os.path.join(experiment_path, experiment_name + '.docx'))


    # batch_size = 1 

    # physical_quantity={
    #     'N':PhysicalQuantity(ranges=[8,15], steps=1, units=None, unit_prefix=None, name='N'),
    #                 }

    # StudySwipe.get_output(study_function=material_sensor,parameter_dict=parameters,physical_quantity=physical_quantity,batch_size=batch_size,output_shape=[len(gamma_ray_doses.values),wavelength.values.shape[-1]])
    # # visualization.visualize(experiment_path, physical_quantity)
    # # ExportUtils.export_images_to_word( experiment_path, physical_quantity, os.path.join(experiment_path, experiment_name + '.docx'))
