import torch
import os
from Methods.TransferMatrixMethod.Wavelength             import WaveLength
from Methods.TransferMatrixMethod.Structure              import Structure
from Methods.TransferMatrixMethod.Layer                  import Layer
from Materials.Materials                                 import Material 
from Methods.TransferMatrixMethod.PhotonicTransferMatrix import PhotonicTransferMatrix
from Study.Study                                         import SaveData, LoadData
from Methods.PhysicalQuantity                            import PhysicalQuantity
from Visualization.Visualization                         import plot

seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

class PhotonicsDataset:
    def __init__(self, N=200):
        self.N = N

    def sample_layer_thickness(self):
        vals = torch.arange(1, 100.5, 0.5)
        layers_thickness = vals[torch.randint(len(vals), (self.N,))]
        return layers_thickness


    def sample_layer_material(self):
        layers_material = torch.randint(0, 2, (self.N, ))
        return layers_material
    
    def structure_transmission(self, wavelength, layers_materials, layers_thickness):
        
    
        SiO2               = Material(wavelength, name="SiO2", refractive_index=1.4618)
        Air                = Material(wavelength, name="Air",  refractive_index=1)

        # Select material based on the sampled value
        if layers_materials == 1:
            layer_mat = SiO2
        else:
            layer_mat = Air

        current_layer = Layer(layer_mat)
        current_layer.thickness = layers_thickness
        
        layers = [current_layer]
        
        structure = Structure(layers, layers_parameters={'method':'repeat', 
                                                                'layers_repeat':[1]})
        
        R, T = PhotonicTransferMatrix.Reflectance_from_layers(structure.layers, theta_0=0, mode='TE')

        outputs = {}
        outputs['Transmission'] = PhysicalQuantity(values=T, units='a.u.', name='Transmission')

        return outputs
    
    def generate_dataset(self, save_dir):
        wavelength = WaveLength(ranges=[150,600], steps=0.005, units="m", unit_prefix="n")
        layers_materials = self.sample_layer_material()
        layers_thickness = self.sample_layer_thickness() * 1e-9 # Convert nm to m
        
        os.makedirs(save_dir, exist_ok=True)

        for i in range(self.N):
            # Run simulation for the i-th sample
            outputs = self.structure_transmission(wavelength, layers_materials[i], layers_thickness[i])
            
            # Plotting
            plot(
                x=wavelength.values,
                y=outputs['Transmission'].values,
                save_dir=save_dir,
                save_name=f'sample_{i}',
                width=256,
                height=256,
                line_colors=['black'],
                line_width=3,
                show=False,
                x_label='Wavelength (nm)',
                y_label='Transmission',
                title=None,
                plot_bgcolor='white',
                template='plotly_white'
            )

