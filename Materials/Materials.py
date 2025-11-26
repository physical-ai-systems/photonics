import torch 
import numpy as np
import os 
import pandas as pd
import torch.nn as nn
from Materials.refractiveindex_sqlite.refractivesqlite import dboperations as refractiveindex_database


class Material(nn.Module):
    def __init__(self,
                 wavelength,
                 name = None,
                 refractive_index = None,
                 permittivity = None,
                 permeability = None,
                 material_index = None,
                 search_refractive_index = None,
                 search_extinction_coefficient = None,
                 only_real = True,
                 get_all_database_properties = False,
                 *args, **kwargs):
        super().__init__()
        
        assert (name) or (refractive_index is not None) or (permittivity is not None and permeability is not None)\
            or (material_index is not None) \
            or (search_refractive_index is not None) or (search_extinction_coefficient is not None)\
            , "name or refractive_index or permittivity and permeability must be defined"

        self.wavelength = wavelength
        self.name = name
        self.refractive_index = refractive_index
        self.permittivity = permittivity
        self.permeability = permeability
        self.material_index = material_index
        self.search_refractive_index = search_refractive_index
        self.search_extinction_coefficient = search_extinction_coefficient
        self.only_real = only_real
        if get_all_database_properties:
            self.get_all_database_properties()
        
        self.device_val = wavelength.values.device if hasattr(wavelength, 'values') else torch.device('cpu')
        self.permittivity_free_space = torch.as_tensor(8.854187817e-12, device=self.device_val)
        self.permeability_free_space = torch.as_tensor(1.2566370614e-6, device=self.device_val)
        path = os.path.dirname(os.path.abspath(__file__))
        self.refractive_index_sqlite_path = os.path.join(path,'refractiveindex_sqlite','refractive.db')


        if refractive_index is not None:
            if isinstance(refractive_index, torch.Tensor):
                assert (wavelength.values.shape[-1] == refractive_index.shape[-1]), "wavelength and refractive_index must have the same shape or be a scalar"
            elif isinstance(refractive_index, (float, int)):
                refractive_index = torch.ones(wavelength.values.shape, device=self.device_val) * refractive_index
            else :
                raise ValueError("refractive_index must be a float, int or torch.Tensor")
            self.refractive_index = refractive_index 


        elif material_index is not None:
            database = refractiveindex_database.Database(self.refractive_index_sqlite_path)
            material = database.get_material(material_index) 
            self.name = material.pageinfo

            # get refractive index -> convert the wavelength to nanometers as the database uses nanometers
            if only_real:
                self.refractive_index = torch.as_tensor([material.get_refractiveindex(wl.item()).item() for wl in (wavelength.values.numpy(force=True)*1e9) ], device=self.device_val)
            else:
                self.refractive_index = torch.as_tensor([material.get_refractiveindex(wl.item()).item() + 1j*material.get_extinctioncoefficient(wl.item()).item() for wl in (wavelength.values.numpy(force=True)*1e9) ], device=self.device_val)

        elif permittivity is not None and permeability is not None:
            if isinstance(permittivity, torch.Tensor):
                assert (wavelength.values.shape[-1] == permittivity.shape[-1]), "wavelength and permittivity must have the same shape or be a scalar"
            elif isinstance(permittivity, (float, int)):
                permittivity = torch.ones(wavelength.values.shape, device=self.device_val) * permittivity
            else :
                raise ValueError("permittivity must be a float, int or torch.Tensor")
            self.permittivity = permittivity

            if isinstance(permeability, torch.Tensor):
                assert (wavelength.values.shape[-1] == permeability.shape[-1]), "wavelength and permeability must have the same shape or be a scalar"
            elif isinstance(permeability, (float, int)):
                permeability = torch.ones(wavelength.values.shape, device=self.device_val) * permeability
            else :
                raise ValueError("permeability must be a float, int or torch.Tensor")
            self.permeability = permeability
        
        elif name is not None:
            raise NotImplementedError            
        elif search_refractive_index is not None:
            raise NotImplementedError
        elif search_extinction_coefficient is not None:
            raise NotImplementedError
        else:
            raise NotImplementedError



        self.get_refractive_index()
        self.get_permittivity()
        self.get_permeability()   
        self.get_average_refractive_index()
        self.get_average_refractive_index_for_real_part()
    
    def to(self, device):
        self.wavelength = self.wavelength.to(device)
        self.refractive_index = self.refractive_index.to(device)
        self.permittivity = self.permittivity.to(device)
        self.permeability = self.permeability.to(device)
        self.permittivity_free_space = self.permittivity_free_space.to(device)
        self.permeability_free_space = self.permeability_free_space.to(device)
        self.refractive_index_avarage = self.refractive_index_avarage.to(device)
        self.refractive_index_avarage_real_part = self.refractive_index_avarage_real_part.to(device)
        return self
    @property
    def device(self):
        return self.refractive_index.device

    def get_refractive_index(self):
        if self.refractive_index is None:
            self.refractive_index = torch.sqrt(self.premitivity * self.permeability) / torch.sqrt(self.permittivity_free_space * self.permeability_free_space)

    def get_average_refractive_index(self):
        self.refractive_index_avarage = torch.mean(self.refractive_index)

    def get_average_refractive_index_for_real_part(self):
        self.refractive_index_avarage_real_part = torch.mean(self.refractive_index.real)
    
    def get_permittivity(self):
        if self.permittivity is None:
            self.permittivity = self.refractive_index**2 
        self.permittivity = self.permittivity * self.permittivity_free_space    
    
    def get_permeability(self):
        if self.permeability is None:
            self.permeability = 1 * torch.ones(self.refractive_index.shape, device=self.refractive_index.device)
        self.permeability = self.permeability * self.permeability_free_space


    def __repr__(self):
        return f'{self.name}'
    
    def __str__(self):
        return f'{self.name}'
    
    def get_all_database_properties(self):
        database = refractiveindex_database.Database(self.refractive_index_sqlite_path)
        all_pages_ids = database._get_all_pageids()
        columns_names = database._get_pages_columns()
        # create a pandas dataframe
        df = pd.DataFrame(columns=columns_names)
        for page_id in all_pages_ids:
            page = pd.DataFrame(database._get_page_info(page_id),index=[0])
            df = pd.concat([df, page], ignore_index=True)
        # save the dataframe to a csv file
        df.to_csv('all_database_properties.csv', index=False)
        return df

class Air(Material):
    def __init__(self, wavelength, *args, **kwargs):
        super().__init__(wavelength, name = 'Air', refractive_index = 1, *args, **kwargs)

class Vacuum(Material):
    def __init__(self, wavelength, *args, **kwargs):
        super().__init__(wavelength, name = 'Vacuum', refractive_index = 1, *args, **kwargs)

class Glass(Material):
    def __init__(self, wavelength, *args, **kwargs):
        super().__init__(wavelength, name = 'Glass', refractive_index = 1.5, *args, **kwargs)

class Silicon(Material):
    def __init__(self, wavelength, *args, **kwargs):
        super().__init__(wavelength, name = 'Silicon', material_index = 'main::SiO2::Silicon', *args, **kwargs)

class Gold(Material):
    def __init__(self, wavelength, *args, **kwargs):
        super().__init__(wavelength, name = 'Gold', material_index = 'main::Au::Gold', *args, **kwargs)

class CompositeMaterial():
    def __init__(self):
        pass
    
    @staticmethod
    def get_material(method=None, materials=None, **kwargs):
        assert method is not None, "method must be specified"
        assert materials is not None, "materials must be specified"
        if method == 'effective_medium_theory':
            return CompositeMaterial.get_effective_medium_theory(method, materials)
        elif method == 'bruggeman':
            return CompositeMaterial.get_bruggeman(method, materials)
        elif method == 'maxwell_garnett':
            return CompositeMaterial.get_maxwell_garnett(method, materials)
        elif method == 'lorentz_lorenz':
            return CompositeMaterial.get_lorentz_lorenz(method, materials)
        elif method == 'drude':
            return CompositeMaterial.get_drude(method, materials)
        elif method == 'drude_lorentz':
            return CompositeMaterial.get_drude_lorentz(method, materials)
        elif method == 'drude_lorentz_anisotropic':
            return CompositeMaterial.get_drude_lorentz_anisotropic(method, materials)
        elif method == 'PorousMedium':
            assert len(materials) == 2, "PorousMedium requires two materials"
            return CompositeMaterial.get_PorousMedium(method, materials, **kwargs)
        elif method == 'Linear':
            assert len(materials) == 2, "Linear requires two materials"
            return CompositeMaterial.get_Linear(method, materials, **kwargs)
        else:
            raise NotImplementedError

    def get_effective_medium_theory(self,method, materials):
        refractive_index = torch.mean(torch.stack([material.refractive_index for material in CompositeMaterial.materials]), dim=0)
        return Material(name=CompositeMaterial.get_name(method, materials),refractive_index = refractive_index)
    
    def get_bruggeman(method, materials, **kwargs):
        raise NotImplementedError

    def get_maxwell_garnett(method, materials, **kwargs):
        raise NotImplementedError

    def get_lorentz_lorenz(method, materials, **kwargs):
        raise NotImplementedError

    def get_drude(method, materials, **kwargs):
        raise NotImplementedError

    def get_drude_lorentz(method, materials, **kwargs):
        raise NotImplementedError

    def get_drude_lorentz_anisotropic(method, materials, **kwargs):
        '''
        Function to calculate the effective refractive index of an anisotropic material.
        based on the Drude-Lorentz model.
        for more information see:
            https://en.wikipedia.org/wiki/Drude_model
            https://en.wikipedia.org/wiki/Lorentz_oscillator
            an example for the air and silicon:
            [Si air]=CompositeMaterial(method=Drude_Lorentz_anisotropic, materials=[Si, air], porosity=0.5)
            This method is taken from the following paper:
            Drude-Lorentz Model for Anisotropic Metamaterials,
            https://doi.org/10.1109/LAWP.2012.2197270
        '''
            
        raise NotImplementedError

    def get_PorousMedium(method, materials, porosity = 0.5, **kwargs):
        '''
        Function to calculate the effective refractive index of a porous medium.
        based on the Bruggeman's equations.
        for more information see:
        https://en.wikipedia.org/wiki/Bruggeman_effective_medium_approximation
        an example for the air and silicon:
        [Si air]=PorousMedium(method=PorousMedium, materials=[Si, air], porosity=0.5) 
        This method is taken from the following paper:

        Bruggeman's equations Z.A. Zaky, A. Sharma, S. Alamri, N. Saleh, A.H. Aly, 
        Detection of Fat Concentration in Milk Using Ternary Photonic Crystal, Silicon, 14 (2021) 6063–6073. 
        https://doi.org/10.1007/s12633-021-01379-8

        an example for the air and silicon:
        [air si]
        epsi = 3 * porosity * (air **2 - si **2) + (2 * si **2 - air **2 );
        n = 0.5 * sqrt(epsi + sqrt( epsi **2 + 8 * si **2 * air **2 ));

        '''
        assert len(materials) == 2, "PorousMedium requires two materials"
        assert materials[0].refractive_index.shape == materials[1].refractive_index.shape, "refractive_index must have the same shape"
        assert torch.all(porosity <= 1) and torch.all(porosity >= 0), "porosity must be between 0 and 1"

        # Check this later
        # refractive_index = 0.5 * torch.sqrt(3 * porosity * (materials[0].refractive_index**2 - materials[1].refractive_index**2) + (2 * materials[1].refractive_index**2 - materials[0].refractive_index**2) + torch.sqrt((3 * porosity * (materials[0].refractive_index**2 - materials[1].refractive_index**2) + (2 * materials[1].refractive_index**2 - materials[0].refractive_index**2))**2 + 8 * materials[1].refractive_index**2 * materials[0].refractive_index**2))
        
        # Material_2.refractive_index  Material_1.refractive_index 
        epsi = 3 * porosity * ( materials[1].refractive_index **2 - materials[0].refractive_index **2 ) + (2 * materials[0].refractive_index **2 - materials[1].refractive_index **2 )
        refractive_index = 0.5 * torch.sqrt( epsi + torch.sqrt( epsi **2 + 8 * materials[0].refractive_index **2 * materials[1].refractive_index **2 ))


        return Material(materials[0].wavelength,name=CompositeMaterial.get_name(method, materials),refractive_index = refractive_index)
    
    def get_Linear(method, materials, Volume_fraction = 0.5, **kwargs):
        '''
        Function to calculate the effective refractive index of a linear composite.'''
        Volume_fraction = torch.as_tensor(Volume_fraction).unsqueeze(-1)
        assert len(materials) == 2, "Linear requires two materials"
        assert materials[0].refractive_index.shape == materials[1].refractive_index.shape, "refractive_index must have the same shape"
        assert torch.all(Volume_fraction <= 1) and torch.all(Volume_fraction >= 0), "Volume_fraction must be between 0 and 1"
        refractive_index = Volume_fraction * materials[0].refractive_index + (1 - Volume_fraction) * materials[1].refractive_index
        return Material(materials[0].wavelength,name=CompositeMaterial.get_name(method, materials),refractive_index = refractive_index)

    def get_name(method, materials):
        return f'{method}({materials[0].name},{materials[1].name})'
        






    

