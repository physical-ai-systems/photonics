import sys 
import os
import time
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha
print(sha)

try:  # we don't need gradients (for now)
    torch._C.set_grad_enabled(False)  # type: ignore
except AttributeError:
    torch._C._set_grad_enabled(False)  # type: ignore
torch.set_default_dtype(torch.float64) 

from Methods.TransferMatrixMethod.Wavelength import WaveLength
from Methods.TransferMatrixMethod.Structure  import Structure
from Methods.TransferMatrixMethod.Layer      import Layer
from Materials.Materials                     import Material 
from Materials.Materials                     import CompositeMaterial
from Methods.TransferMatrixMethod.AnnularPhotonicTransferMatrix import AnnularPhotonicTransferMatrix
from Materials.Materials_database.Read_material import Read_material
from Visualization.Visualization             import plot
from Study.OutputAnalysis                   import Peak_analysis
from Study.Study                   import Study_parameter

def organic_material_sensor(wavelength, experiment_path, method, peak_analysis,
                            N, rho0, porosity_1, porosity_2,
                            layer_1_thickness, layer_2_thickness, defect_thickness,
                            Volume_fraction, plot_flag = False, **kwargs):

    # Define the materials
    Si  = Material(wavelength, name="Si", refractive_index=3.5)
    SiO2 = Material(wavelength, name="SiO2", refractive_index=1.45)
    Air          = Material(wavelength, name="Air", refractive_index=1)
    Water        = Material(wavelength, name="Water", refractive_index=1.33230545)
    E_coli       = Material(wavelength, name="E_coli", refractive_index=1.39)

    # Define the materials
    variable_material = CompositeMaterial.get_material('Linear', [E_coli,Water], Volume_fraction = Volume_fraction)
    if plot_flag:
        plot(x=wavelength.values * 1e9, y=variable_material.refractive_index, names=[None], title=None, x_label='Wavelength (nm)', y_label='n (Arbitrary units)', save_dir=experiment_path, save_name='Refactive index', show=True)

    material_porose_1 = CompositeMaterial.get_material('PorousMedium', [variable_material,Si], porosity=porosity_1)
    material_porose_2 = CompositeMaterial.get_material('PorousMedium', [variable_material,Si], porosity=porosity_2)

    # Define the layers
    layer0 = Layer(SiO2,thickness=rho0*1e-9)
    layerd = Layer(variable_material, thickness=defect_thickness*1e-9)
    layer1 = Layer(material_porose_1, thickness=layer_1_thickness*1e-9)
    layer2 = Layer(material_porose_2, thickness=layer_2_thickness*1e-9)
    layerf = Layer(SiO2, thickness=1000e-9)
    # print(layer0.thickness, layer1.thickness, layer2.thickness, layerd.thickness, layerf.thickness)
    # Define the structure
    structure = Structure([layer0, [layer1, layer2], layerd ,[layer1, layer2],layerf],
                                        layers_parameters = {'method':'repeat', 
                                                                'layers_repeat':[1,N,1,N,1],
                                                                })

    # # get the reflectance
    Reflectance = method.Reflectance_from_layers(structure.layers, m=0, mode= 'TE')
    output = peak_analysis.defect_peak_analysis(wavelength.get_in_unit(),Reflectance,wavelength = wavelength,  structure_type='periodic_with_defect', number_of_defect_peaks=1, refractive_index = variable_material.refractive_index)

    # points= peak_analysis.convert_defect_peak_out_dict_to_indicies(output)
    if plot_flag:
        points = peak_analysis.convert_dicts_into_signle_dict(output)
        points = None
        plot(x=wavelength.get_in_unit(), y=[Reflectance], highlight_points=[points],names=[None], title=None, x_label='Wavelength (nm)', y_label='Reflectance (Arbitrary units)', save_dir=experiment_path, save_name='Reflectance', show=True)
    return Reflectance, output

if __name__ == "__main__":
    # Define the experimental setup
    experiment_name        = os.path.basename(__file__).split('.')[0]
    experiment_description = "This experiment ..."
    experiment_author      = "Ayman A. Ameen"
    experiment_date        = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    experiment_path        = os.path.join(os.path.dirname(__file__), experiment_name) # experiment_path        = os.path.join(os.path.dirname(__file__), 'Experiments', experiment_name + '_' + experiment_date)

    study_description = {'experiment_name':experiment_name, 'experiment_description':experiment_description, 'experiment_author':experiment_author, 'experiment_date':experiment_date, }

    # Define the method
    method = AnnularPhotonicTransferMatrix()
    peak_analysis = Peak_analysis(input_type='Reflectance', input_total='without_absorption')
    # Define the wavelength range
    wavelength = WaveLength(wavelength_range=[450,750],wavelength_step=0.005, wavelength_range_unit="nm")
    N    = 10
    rho0 = 500 # nm
    porosity_1        =  0.7
    porosity_2        =  0.4
    layer_1_thickness = 50 # nm
    layer_2_thickness = 65 # nm
    defect_thickness  = 48 # nm
    Volume_fraction   = 0.4 
    plot_flag         = False #True # False
    parameters = {'wavelength':wavelength, 'experiment_path':experiment_path, 'method':method, 'peak_analysis':peak_analysis, 
                  'N':N, 'rho0':rho0, 'porosity_1':porosity_1,'porosity_2':porosity_2,'layer_1_thickness':layer_1_thickness, 'layer_2_thickness':layer_2_thickness, 'defect_thickness':defect_thickness, 
                   'Volume_fraction':Volume_fraction, 'plot_flag':plot_flag }

    # First plot the reflectance and reflective index 
    Reflectance, output = organic_material_sensor(**parameters)

    study_parameters_all   = [ 'Volume_fraction', 'defect_thickness', 'rho0',                  'N']
    study_parameters_name  = [ 'Volume fraction', 'Defect thickness', 'Inner core radius',     'N']
    study_parameters_unit  = [              None,               'nm',                'nm',      None]
    study_parameters_range = [            [0, 1],           [30, 80],          [200, 700],  [8,15]]
    study_parameters_step  = [              0.05,                  2,                 10,       1 ]
    reference_index        = [                 0,                  0,                   0,      0 ]
    
    
    
    
    # study_parameters_all   = [ 'layer_1_thickness', 'layer_2_thickness', 'porosity_1',       'porosity_2']
    # study_parameters_name  = [ 'Layer a thickness', 'Layer b thickness', 'Layer a porosity', 'Layer b porosity']
    # study_parameters_unit  = [                'nm',                'nm',               None,               None]
    # study_parameters_range = [            [45, 55],            [60, 70],          [0.6,0.8],         [0.3, 0.5]]
    # study_parameters_step  = [                 0.2,                 0.2,               0.005,             0.005]
    # reference_index        = [                   0,                   0,                   0,                0 ]



# pylint: disable=consider-using-enumerate
    for counter in range(len(study_parameters_all)):
        study_parameters = Study_parameter(study_function=organic_material_sensor,
                                        parameter_dict=parameters,
                                        study_parameters=[study_parameters_all[counter],],
                                        study_parameters_name = study_parameters_name[counter],
                                        study_parameters_unit=study_parameters_unit[counter],
                                        study_parameters_range=study_parameters_range[counter],
                                        study_parameters_step= study_parameters_step[counter],
                                        study_name='organic_material_sensor',
                                        reference_index= reference_index[counter],
                                        study_description=study_description,).get_study()



