import numpy as np
import torch
from Methods.PhysicalQuantity import PhysicalQuantity

class WaveLength(PhysicalQuantity):
    def __init__(self, values=None, units='m', ranges=[300,500], steps=1, unit_prefix='n', name='Wavelength', center=None, shift_from_center=None, **kwargs):
        super().__init__(values=values, units=units, ranges=ranges, steps=steps, unit_prefix=unit_prefix, name=name, center=center, shift_from_center=shift_from_center, **kwargs)
        self.speed_of_light = 299792458
        self.omega = 2 * np.pi * self.speed_of_light / self.values
        self.k     = 2 * np.pi / self.values
    def broadcast(self, shape):
        self.values = torch.broadcast_to(self.values, shape)
        self.omega  = torch.broadcast_to(self.omega, shape)
        self.k      = torch.broadcast_to(self.k, shape)

    def to(self, device):
        self.values = self.values.to(device)
        self.omega  = self.omega.to(device)
        self.k      = self.k.to(device)
        return self

# class WaveLength:
#     def __init__(self, wavelength_range=[300,500], wavelength_step=1, wavelength_range_unit="nm", wavelength_scale=1e-9, wavelength_center=None, wavelength_shfit_from_center=None,wavelength_unit='m', *args, **kwargs):

#         self.wavelength_range      = wavelength_range
#         self.wavelength_step       = wavelength_step
#         self.wavelength_range_unit = wavelength_range_unit
#         self.wavelength_unit       = wavelength_unit
#         self.wavelength_scale      = wavelength_scale
#         self.wavelength_center     = wavelength_center
#         self.wavelength_shfitrom_center = wavelength_shfit_from_center
#         self.speed_of_light = 299792458
#         self.get_and_check_wavelenth_scale_factor()

#         self.values = torch.arange(wavelength_range[0], wavelength_range[1], wavelength_step) * wavelength_scale

#         if wavelength_center is None:
#             self.wavelength_center = (wavelength_range[0]+wavelength_range[1]) / (wavelength_shfit_from_center if wavelength_shfit_from_center is not None else 2)
#             self.wavelength_center = self.wavelength_center * wavelength_scale

#         self.omega = 2 * np.pi * self.speed_of_light / self.values
#         self.k     = 2 * np.pi / self.values


#     def get_and_check_wavelenth_scale_factor(self,):
#         try :
#             wavelength_scale = self.unit_conversion(self.wavelength_range_unit, self.wavelength_unit)
#         except:
#             Warning("Can't check wavelength scale factor for unit {}".format(self.wavelength_range_unit ))
                 
#         if self.wavelength_scale is None:
#             self.wavelength_scale = wavelength_scale
#         elif self.wavelength_scale != wavelength_scale:
#             raise ValueError("Wavelength scale {} is not equal to {}".format(self.wavelength_scale, wavelength_scale))
#         else:
#             pass

#     def unit_conversion(self, input_unit, output_unit):
#         '''
#         This function converts the input unit to the output unit.
#         To do this more easily, the input unit are converted to the base unit and then converted to the output unit.
#         '''       
#         length_units = ['km', 'm', 'dm', 'cm', 'mm', 'um', 'nm', 'pm', 'fm', 'am', 'zm']
#         frequency_units = ['Hz', 'kHz', 'MHz', 'GHz', 'THz', 'PHz', 'EHz', 'ZHz']
#         scale_factor_length_to_m = [1e3, 1, 1e-1, 1e-2, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21]
#         scale_factor_frequency_to_Hz = [1, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21]
#         factor_Hz_to_m = 1 / self.speed_of_light
#         factor_m_to_Hz = 1 / self.speed_of_light

#         if input_unit == output_unit:
#             return 1
#         elif input_unit in length_units and output_unit in length_units:
#             return scale_factor_length_to_m[length_units.index(input_unit)] / scale_factor_length_to_m[length_units.index(output_unit)]
#         elif input_unit in frequency_units and output_unit in frequency_units:
#             return scale_factor_frequency_to_Hz[frequency_units.index(input_unit)] / scale_factor_frequency_to_Hz[frequency_units.index(output_unit)]
#         elif input_unit in length_units and output_unit in frequency_units:
#             return scale_factor_length_to_m[length_units.index(input_unit)] * factor_m_to_Hz / scale_factor_frequency_to_Hz[frequency_units.index(output_unit)]
#         elif input_unit in frequency_units and output_unit in length_units:
#             return scale_factor_frequency_to_Hz[frequency_units.index(input_unit)] * factor_Hz_to_m / scale_factor_length_to_m[length_units.index(output_unit)]
#         else:
#             raise ValueError("Can't convert {} to {}".format(input_unit, output_unit))



#     def get_in_unit(self, unit=None):
#         if unit is None:
#             unit = self.wavelength_range_unit
#         else:
#             pass
#         return self.values.numpy(force=True) * self.unit_conversion(self.wavelength_unit, unit)

#     def convert_input_to_wavelength_unit(self, input):
#         return input * self.unit_conversion(self.wavelength_unit, self.wavelength_range_unit)

#     def get_wavelength_range(self):
#         return self.wavelength_range

#     def get_wavelength_step(self):
#         return self.wavelength_step

#     def get_wavelength_unit(self):
#         return self.wavelength_unit

#     def get_wavelength_scale(self):
#         return self.wavelength_scale

#     def get_wavelength_center(self):
#         return self.wavelength_center

#     def get_wavelength_shift_from_center(self):
#         return self.wavelength_shfitrom_center

#     def get_omega(self):
#         return self.omega

#     def get_k(self):
#         return self.k



#     def __repr__(self):
#         return "Wavelength range: {}, step: {}, unit: {}, scale: {}, center: {}, shift from center: {}".format(self.wavelength_range, self.wavelength_step, self.wavelength_unit, self.wavelength_scale, self.wavelength_center, self.wavelength_shfitrom_center)
    
#     def __str__(self):
#         return "Wavelength range: {}, step: {}, unit: {}, scale: {}, center: {}, shift from center: {}".format(self.wavelength_range, self.wavelength_step, self.wavelength_unit, self.wavelength_scale, self.wavelength_center, self.wavelength_shfitrom_center)
    
#     def __len__(self):
#         return len(self.values)
    
#     def __getitem__(self, key):
#         return self.values[key]
    
#     def __iter__(self):
#         return iter(self.values)
    
#     def __next__(self):
#         return next(self.values)
    
#     def __add__(self, other):
#         return self.values + other
    
#     def __sub__(self, other):
#         return self.values - other
    
#     def __mul__(self, other):
#         return self.values * other
    
#     def __truediv__(self, other):
#         return self.values / other
    
#     def __floordiv__(self, other):
#         return self.values // other
    
#     def __mod__(self, other):
#         return self.values % other
    
#     def __pow__(self, other):
#         return self.values ** other
    
#     def __lshift__(self, other):
#         return self.values << other
    
#     def __rshift__(self, other):
#         return self.values >> other
    
#     def __and__(self, other):
#         return self.values & other
    
#     def __xor__(self, other):
#         return self.values ^ other
    
#     def __or__(self, other):
#         return self.values | other
    
#     def __lt__(self, other):
#         return self.values < other
    
#     def __le__(self, other):
#         return self.values <= other
    
#     def __eq__(self, other):
#         return self.values == other
    
#     def __ne__(self, other):
#         return self.values != other
    
#     def __gt__(self, other):
#         return self.values > other
    
#     def __ge__(self, other):
#         return self.values >= other
    
#     def __neg__(self):
#         return -self.values
    
#     def __pos__(self):
#         return +self.values
    



