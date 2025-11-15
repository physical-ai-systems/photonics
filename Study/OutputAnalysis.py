import torch 
import numpy as np
import scipy 
from typing import Literal
from Methods.PhysicalQuantity import PhysicalQuantity
from Utils.Utils import arange_like, unsqueeze_physical_properties
from Visualization.Visualization             import plot
from functools import partial
import copy
import traceback


class Analysis():
    def get_output(self, wavelength=None,
                    T=None, R=None, A=None, Intensity=PhysicalQuantity(values=[1], name='Intensity', units='a.u.'),
                    PBG_range=None,
                    property_value=None, working_dim=None, referance_index=None,
                    backend='scipy', strcuture_type='periodic_with_defect', number_of_defect_peaks=1,):

        assert wavelength is not None, "The wavelength is not provided"
        assert T is not None or R is not None, "The input is not provided"

        T, R, A = self.get_T_R_A(T, R, A, Intensity)

        input_anaysis = {'wavelength':wavelength, 'T':T, 'R':R, 'A':A, 'property_value':property_value, 'PBG_range':PBG_range, 'working_dim':working_dim, 'referance_index':referance_index, 'backend':backend, 'strcuture_type':strcuture_type, 'number_of_defect_peaks':number_of_defect_peaks,}
        output = {}

        if backend == 'scipy':
              output.update(self.get_output_scipy(input_anaysis))
        elif backend == 'torch':
            output.update(self.get_output_torch(input_anaysis))

        if strcuture_type == 'periodic_with_defect':    
            output.update(self.get_QualityFactor(output['peak_wavelength'], output['FullWidthHalfMax']))

        if property_value is not None:            
            # sensitivity_parameter = output['peak_wavelength'] if strcuture_type == 'periodic_with_defect' else output['band_gap_center']
            sensitivity_parameter = output['peak_wavelength'] if strcuture_type == 'periodic_with_defect' else output['band_gap_begining']
            figure_of_merit_parameter = output['FullWidthHalfMax'] if strcuture_type == 'periodic_with_defect' else output['band_gap_width']

            sensitivity_parameter_values, sensitivity_parameter_referance = self.separate_values_from_reference(sensitivity_parameter.values, referance_index, working_dim)  
            figure_of_merit_parameter_values, figure_of_merit_parameter_referance = self.separate_values_from_reference(figure_of_merit_parameter.values, referance_index, working_dim)

            property_values = self.get_value(property_value.values, wavelength.values, sensitivity_parameter.values.unsqueeze(dim=-1))
            property_values, property_referance = self.separate_values_from_reference(property_values, referance_index, working_dim)

            Sensitivity = self.get_Sensitivity(sensitivity_parameter_values, property_values, sensitivity_parameter_referance, property_referance, units= sensitivity_parameter.units+'/'+property_value.units)
            FigureOfMerit = self.get_FigureOfMerit(Sensitivity.values, figure_of_merit_parameter_values, property_value)

            output.update({'Sensitivity':Sensitivity, 'FigureOfMerit':FigureOfMerit})

        return output

    def get_T_R_A(self, T=None, R=None, A=None, Intensity: PhysicalQuantity = PhysicalQuantity(values=[1], name='Intensity', units='a.u.')):

        assert T is not None or R is not None, "The input is not provided"

        if A is None and T is not None and R is not None:
                A = PhysicalQuantity(Intensity - R.values - T.values)
        else:
            A = PhysicalQuantity(torch.tensor(0))

        if T is None:
            T = PhysicalQuantity(Intensity - R.values  - A.values)
        elif R is None:
            R = PhysicalQuantity(Intensity - T.values  - A.values)

        return T, R, A

    def find_peaks(self, y, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
        peaks, peaks_properties = scipy.signal.find_peaks(y, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
        return peaks, peaks_properties

    def get_FullWidthHalfMax_scipy(self, y, peaks, rel_height=0.5):
        peak_widths = scipy.signal.peak_widths(y, peaks, rel_height=rel_height)
        FullWidthHalfMax, index_left, index_right = [peak_widths[0][counter] for counter in range(len(peaks))], [int(peak_widths[2][counter]) for counter in range(len(peaks))], [int(peak_widths[3][counter]) for counter in range(len(peaks))]
        return FullWidthHalfMax, index_left, index_right
    
    def get_value(self, values, reference_values, reference_value): 
        reference_indices = torch.argmin(torch.abs(reference_values - reference_value),dim=-1,keepdim=True)
        reference_indices_indices = list(range(len(reference_indices.shape)))
        reference_indices = reference_indices.permute(reference_indices_indices[-1::-1])
        return torch.gather(values, dim=-1, index=reference_indices).permute(reference_indices_indices[-1::-1]).squeeze()

    def get_values_in_range(self, values, reference_values, range=None):
        additional_index = 0
        values_original = copy.deepcopy(values)
        if range is not None and range[0] != 0 and range[1] != 0:
            range = [np.argmin(np.abs(reference_values - range[0])), np.argmin(np.abs(reference_values - range[1]))]
            values = values[range[0]:range[1]]
            additional_index = range[0]
        return values, additional_index, values_original


    def get_peak_sorted(self, y, height=None, threshold=None, distance=None, prominence=0.1, width=None, wlen=None, rel_height=0.5, plateau_size=None):
        peaks, peaks_properties = self.find_peaks(y, height=height, threshold=threshold, distance=distance, prominence=prominence, width=width, wlen=wlen, rel_height=rel_height, plateau_size=plateau_size)
        FullWidthHalfMax, index_left, index_right = self.get_FullWidthHalfMax_scipy(y, peaks)
        peak_sorted_prominences = peaks_properties['prominences'].argsort()
        peak_sorted_width       = np.array(FullWidthHalfMax).argsort()
        return peaks, peaks_properties, FullWidthHalfMax, peak_sorted_prominences, peak_sorted_width

    def get_band_gap(self,
        R,
        wavelength,
        structure_type: Literal['periodic', 'periodic_with_defect'] = 'periodic_with_defect',
        number_of_defect_peaks: int = 1,
        PBG_range = None,
        mode = 'prominence',
        ):               

        try:
            R, Additional_index, R_original = self.get_values_in_range(R, wavelength, PBG_range) # get the values in the range

            if structure_type == 'periodic':  # if the structure is periodic, then we get the most prominent peak as the band gap and return its begining and end and width
                number_of_defect_peaks = 0
                
            height = np.array([0.7,1])
            peaks, peaks_properties, FullWidthHalfMax, peak_sorted_prominences, peak_sorted_width = self.get_peak_sorted(R, height=height)
            # if the structure is periodic with defect, then get the get the widest peak as the main part of the band gap and pick the widest and the most prominent peaks next to it as the rest of the band gap peaks
            for i in range(number_of_defect_peaks+1):
                if i == 0:
                    height = np.array([0.99,1])
                    peaks_first, _, _, peak_sorted_prominences_first, peak_sorted_width_first = self.get_peak_sorted(R, height=height)
                    if mode == 'FWHM':
                        peak_indices = np.array([peaks.tolist().index(
                            # peaks_first[peak_sorted_width_first[-1]]
                            peaks_first[0]
                            )])
                    elif mode == 'prominence':
                        peak_indices = np.array(peaks.tolist().index(
                            # peaks_first[peak_sorted_prominences_first[-1]]
                            peaks_first[0]
                            ))

                else:
                    possible_peak_indices = np.append(peak_indices+1, peak_indices-1)
                    possible_peak_indices = [index for index in possible_peak_indices if index >= 0 and index < len(peaks)]
                    # get the max and min of the possible peaks
                    possible_peak_indices =[max(possible_peak_indices), min(possible_peak_indices)]
                    # get the most prominent peak from the possible peaks
                    peak_indices_prominences = np.array(max(possible_peak_indices, key=lambda index: peaks_properties['prominences'][index]))
                    peak_indices_FWHM        = np.array(max(possible_peak_indices, key=lambda index: FullWidthHalfMax[index]))

                    if mode == 'FWHM':
                        peak_indices = np.append(peak_indices,peak_indices_FWHM)
                    elif mode == 'prominence':
                        peak_indices = np.append(peak_indices,peak_indices_prominences)
                    else:   
                        raise ValueError("The mode is not implemented yet")


            
            _ , index_left, index_right = self.get_FullWidthHalfMax_scipy(R, peaks[peak_indices] if len(peak_indices.shape)>0 else np.array([peaks[peak_indices]])) # get the FullWidthHalfMax and the left and right index of the defect peaks 
            band_gap_begining, band_gap_end = int(np.min(index_left)), int(np.max(index_right)) #Get the lowest left index and the highest right index
            band_gap_begining, band_gap_end = band_gap_begining + Additional_index, band_gap_end + Additional_index # add the additional index to the band gap begining and end

            band_gap_begining, band_gap_end = wavelength[band_gap_begining], wavelength[band_gap_end] # convert the band gap begining and end to the wavelength
            band_gap_width    = band_gap_end - band_gap_begining
            band_gap_center   = (band_gap_begining + band_gap_end)/2

            average_peak_intensity = torch.as_tensor(np.mean(R[peaks[peak_indices]]))
        except Exception as e:
            print('The band gap analysis failed with the following error: ', e)
            traceback.print_exc()
            band_gap_begining, band_gap_end, band_gap_width, band_gap_center, average_peak_intensity = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        return {'band_gap_begining':band_gap_begining, 'band_gap_end':band_gap_end, 'band_gap_width':band_gap_width, 'band_gap_center':band_gap_center, 'average_peak_intensity':average_peak_intensity}

    def to_numpy(self, values=dict):
        for key in values.keys():
            if type(values[key]) == torch.Tensor:
                values[key] = values[key].cpu().numpy()
            elif type(values[key]) == PhysicalQuantity:
                values[key] = values[key].values.cpu().numpy()
        return values

    def to_torch(self, values=dict):
        for key in values.keys():
            if type(values[key]) == np.ndarray:
                values[key] = torch.tensor(values[key])
            elif type(values[key]) == PhysicalQuantity:
                values[key] = torch.tensor(values[key].values)
        return values

    def reshape_values_last_dim_numpy(self, values=dict,):
        for key in values.keys():
            if type(values[key]) == np.ndarray:
                original_shape = values[key].shape[:-1]
                values[key] = values[key].reshape(-1, values[key].shape[-1])

        return values, original_shape

    def loop_over_values(self, values:dict, function:callable, **kwargs):
        keys = list(values.keys())
        for i in range(values[keys[0]].shape[0]):
            print(i)
            if i == 0:
                output = function(**{key:values[key][i,...].squeeze() for key in keys}, **kwargs)
                output = {key:output[key].unsqueeze(0) for key in output.keys()}
            else:
                for key in output.keys():
                    output[key] = torch.cat([
                    output[key],
                    function(**{key:values[key][i,...].squeeze() for key in keys}, **kwargs)[key].unsqueeze(0)
                    ], axis=0)
        return output

    def convert_to_physical_quantity(self, values:dict, wavelength:PhysicalQuantity, R:PhysicalQuantity, T:PhysicalQuantity, original_shape):
        names = {'band_gap_begining' : 'PBG left edge', 'band_gap_end' : 'PBG right edge',
                 'band_gap_width' : 'PBG width', 'band_gap_center' : 'PBG center',
                 'average_peak_intensity' : 'Average peak intensity',
                 'peak_wavelength' : 'Defect peaks wavelength', 'defect_intensity_T' : 'Defect peaks intensity',
                 'defect_intensity_R' : 'Defect peaks intensity', 'FullWidthHalfMax' : 'Full Width Half Max',
                 'position_left_defect' : 'Left defect position', 'position_right_defect' : 'Right defect position',
                 'defect_peak_width' : 'Defect peak width',
        }
        units = {'band_gap_begining' : wavelength.units, 'band_gap_end' : wavelength.units,
                'band_gap_width' : wavelength.units, 'band_gap_center' : wavelength.units,
                'average_peak_intensity' : R.units,
                'peak_wavelength' : wavelength.units, 'defect_intensity_T' : T.units,
                'defect_intensity_R' : R.units, 'FullWidthHalfMax' : wavelength.units,
                'position_left_defect' : wavelength.units, 'position_right_defect' : wavelength.units,
                'defect_peak_width' : wavelength.units,
            }

        unit_prefix = {'band_gap_begining' : wavelength.unit_prefix, 'band_gap_end' : wavelength.unit_prefix,
                'band_gap_width' : wavelength.unit_prefix, 'band_gap_center' : wavelength.unit_prefix,
                'average_peak_intensity' : R.unit_prefix,
                'peak_wavelength' : wavelength.unit_prefix, 'defect_intensity_T' : T.unit_prefix,
                'defect_intensity_R' : R.unit_prefix, 'FullWidthHalfMax' : wavelength.unit_prefix,
                'position_left_defect' : wavelength.unit_prefix, 'position_right_defect' : wavelength.unit_prefix,
                'defect_peak_width' : wavelength.unit_prefix,
            }
        for key in values.keys():
            values[key] = PhysicalQuantity(values=values[key].reshape(original_shape), name=names[key], units=units[key], unit_prefix=unit_prefix[key])
        return values

    def get_output_scipy(self, input_analysis):
        Values, original_shape = self.reshape_values_last_dim_numpy(self.to_numpy({'R':input_analysis['R'], 'T':input_analysis['T'], 'PBG_range':input_analysis['PBG_range']}))
        function = partial(self.get_band_gap, wavelength=input_analysis['wavelength'][...,:], structure_type=input_analysis['strcuture_type'], number_of_defect_peaks=input_analysis['number_of_defect_peaks'], PBG_range=input_analysis['PBG_range'])
        output = self.loop_over_values({'R':Values['R'],}, function)

        if input_analysis['strcuture_type'] == 'periodic_with_defect':
            function = partial(self.defect_peak_analysis, wavelength=input_analysis['wavelength'][...,:], number_of_defect_peaks=input_analysis['number_of_defect_peaks'])
            output.update(self.loop_over_values({'T':Values['T'], 'R':Values['R'], 'PBG_range': torch.cat([output['band_gap_begining'].unsqueeze(-1), output['band_gap_end'].unsqueeze(-1)], dim=-1),},
             function))
        
        output = self.convert_to_physical_quantity(output, input_analysis['wavelength'], input_analysis['R'], input_analysis['T'], original_shape)
        return output
        
    def defect_peak_analysis(self,
                            T,
                            R,
                            wavelength,
                            PBG_range = None,
                            number_of_defect_peaks: int = 1,
                            ): 
        try:
            T, Additional_index, T_original = self.get_values_in_range(T, wavelength, PBG_range) # get the values in the range
            peaks, peaks_properties = self.find_peaks(T) # get the peaks and their properties
            if number_of_defect_peaks != len(peaks):
                print(f"The number of defect peaks {number_of_defect_peaks} is not equal to the number of peaks {len(peaks)}")
                # select the most prominent peaks
                peaks = peaks[peaks_properties['prominences'].argsort()[-number_of_defect_peaks:]]
            FullWidthHalfMax, index_left_defect, index_right_defect = self.get_FullWidthHalfMax_scipy(T, peaks) # get the FullWidthHalfMax and the left and right index of the defect peaks
            FullWidthHalfMax, index_left_defect, index_right_defect = torch.as_tensor(FullWidthHalfMax), torch.as_tensor(index_left_defect), torch.as_tensor(index_right_defect)
            defect_intensity_T = torch.as_tensor(T[peaks])
            defect_intensity_R = torch.as_tensor(R[peaks+Additional_index.item()])
            peaks, position_left_defect, position_right_defect = wavelength[peaks + Additional_index.item()], wavelength[index_left_defect+ Additional_index], wavelength[index_right_defect + Additional_index]
            defect_peak_width = position_right_defect - position_left_defect
            FullWidthHalfMax = FullWidthHalfMax * defect_peak_width / (index_right_defect - index_left_defect)
        except Exception as e:
            print('The defect peak analysis failed with the following error: ', e)
            traceback.print_exc()
            peaks, position_left_defect, position_right_defect, FullWidthHalfMax, defect_intensity_T, defect_intensity_R, defect_peak_width = torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0]), torch.tensor([0])
        return {'peak_wavelength':peaks, 'position_left_defect':position_left_defect, 'position_right_defect':position_right_defect, 'FullWidthHalfMax':FullWidthHalfMax, 'defect_intensity_T':defect_intensity_T, 'defect_intensity_R':defect_intensity_R, 'defect_peak_width':defect_peak_width}
        

    
    def get_QualityFactor(self, peak_wavelength, FullWidthHalfMax):
        QualityFactor = peak_wavelength.values/FullWidthHalfMax.values
        return {'QualityFactor':PhysicalQuantity(values=QualityFactor, name='Quality Factor')}
    
    def get_Sensitivity(self, property, property_value, property_dash, property_dash_value, units=None):
        Sensitivity = (property - property_dash)/(property_value - property_dash_value)
        return PhysicalQuantity(values=Sensitivity, name='Sensitivity', units=units)
    
    def get_FigureOfMerit(self, Sensitivity, FullWidthHalfMax, property_value):
        FigureOfMerit = Sensitivity/FullWidthHalfMax
        return PhysicalQuantity(values=FigureOfMerit, name='Figure of Merit', units= '1/' + property_value.units )
    
    def separate_values_from_reference(self, values, reference_index = 0, working_dim = -1):

        assert reference_index is not None or working_dim is not None, "The reference index and dimension are not provided"

        if working_dim is not None:
            if working_dim == -1:
                working_dim = len(values.shape) - 1
            elif working_dim >= len(values.shape):
                raise ValueError("The wroking dimension {} is out of range".format(working_dim))
            logical_idx = arange_like(values, working_dim) == reference_index
            values_reference = values[logical_idx]
            values_rest      = values[~logical_idx]
        else:
            raise ValueError("The working dimension is not provided")


        return values_rest, values_reference  

    def get_output_torch(self,output):
        Warning("The output is not testet for torch")
        wavelength, T, R, A, PBG_range = output['wavelength'], output['T'], output['R'], output['A'], output['PBG_range']
        assert PBG_range is not None, "The PBG range is not provided and get_PBG is not implemented for torch"
        
        logical_idx, peak_index, peak_wavelength, peak_intensity_T, peak_intensity_R = self.get_peak_torch(wavelength, T, R, PBG_range)
        FullWidthHalfMax = self.get_FullWidthHalfMax_torch(peak_index, T, wavelength, logical_idx)

        return {'peak_wavelength':peak_wavelength, 'peak_intensity_T':peak_intensity_T, 'peak_intensity_R':peak_intensity_R, 'FullWidthHalfMax':FullWidthHalfMax}


    def get_peak_torch(self, wavelength, T, R, PBG_range,):
        logical_idx       = torch.logical_and(wavelength > PBG_range[...,0], wavelength < PBG_range[...,1])
        peak_index        = torch.argmax(torch.where(logical_idx, T.values, 0), dim=-1, keepdim=True)
        peak_wavelength   = PhysicalQuantity(values=wavelength.values.gather(-1, peak_index).squeeze(), units=wavelength.units, name='Defect peaks wavelength')
        peak_intensity_T  = PhysicalQuantity(values=T.values.gather(-1, peak_index).squeeze(),  name='Defect peaks intensity')
        peak_intensity_R  = PhysicalQuantity(values=R.values.gather(-1, peak_index).squeeze(),  name='Defect peaks intensity')
        return logical_idx, peak_index, peak_wavelength, peak_intensity_T, peak_intensity_R


    def get_FullWidthHalfMax_torch(self, peak_index, T, wavelength, logical_idx):
        # arange_index = torch.arange(T.values.shape[-1]).view(*(len(T.values.shape)-1)*[1],-1).repeat(*[*T.values.shape[:-1],1])
        arange_index     =  arange_like(T.values, -1)
        HalfMax          = T.values.gather(-1, peak_index)/2
        index_left       = torch.argmin(torch.abs(torch.where(torch.logical_and(arange_index<peak_index, logical_idx),T.values,0) - HalfMax),dim=-1, keepdim=True)
        index_right      = torch.argmin(torch.abs(torch.where(torch.logical_and(arange_index>peak_index, logical_idx),T.values,0) - HalfMax),dim=-1, keepdim=True)
        FullWidthHalfMax = wavelength.values.gather(-1,index_right) - wavelength.values.gather(-1,index_left)  
        return PhysicalQuantity(values=FullWidthHalfMax.squeeze(), units=wavelength.units, name='Full Width Half Max')        



# OLD CODE

    # def get_output_scipy(self, output):
    #     # convert the PBG from the wavelength to the index
    #     output_new = self.recursive_output(output)

    #     # Convert all of the outputs to wavelength in form in physical quantity
    #     output_new['band_gap_begining'] = PhysicalQuantity(values=output_new['band_gap_begining'].squeeze(), name='PBG left edge', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     output_new['band_gap_end']      = PhysicalQuantity(values=output_new['band_gap_end'].squeeze(), name='PBG right edge', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     output_new['band_gap_width']    = PhysicalQuantity(values=output_new['band_gap_width'].squeeze(), name='PBG width', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     output_new['band_gap_center']   = PhysicalQuantity(values=output_new['band_gap_center'].squeeze(), name='PBG center', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     try:
    #         output_new['peak_wavelength']   = PhysicalQuantity(values=output_new['peak_wavelength'].squeeze(), name='Defect peaks wavelength', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #         output_new['peak_intensity_T']  = PhysicalQuantity(values=output_new['peak_intensity_T'].squeeze(), name='Defect peaks intensity', units=output['T'].units, unit_prefix = output['T'].unit_prefix)
    #         output_new['peak_intensity_R']  = PhysicalQuantity(values=output_new['peak_intensity_R'].squeeze(), name='Defect peaks intensity', units=output['R'].units, unit_prefix = output['R'].unit_prefix)
    #         output_new['FullWidthHalfMax']  = PhysicalQuantity(values=output_new['FullWidthHalfMax'].squeeze(), name='Full Width Half Max', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     except Exception as e:
    #         print('The defect peak analysis failed with the following error: ', e)
    #     output_new['PBG_range']         = PhysicalQuantity(values=output_new['PBG_range'].squeeze(), name='PBG range', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)
    #     output_new['PBG_range_addition'] = PhysicalQuantity(values=output_new['PBG_range_addition'].squeeze(), name='PBG range addition', units=output['wavelength'].units, unit_prefix = output['wavelength'].unit_prefix)


    #     return output_new

    # def recursive_output(self, output):
    #     # TODO : Just reshape the output[..., last dim] and iterate over the first dimension to get the output
    #     # create new output which contains the following:
    #     # if it pytorch then recursively call the function with the new output until the dim = 1 
    #     var_dim = output['T']
    #     keys_to_iterate = ['wavelength', 'T', 'R', 'PBG_range','PBG_range_addition']
    #     dim_num = len(var_dim.shape)
    #     target_var_dim = var_dim.shape[0]
    #     if dim_num == 1: 
    #         output_new = self.get_single_output_scipy(output)
    #         return output_new
            
    #     elif dim_num > 1:
    #         for i in range(target_var_dim):
    #             input_new = {}
    #             for key in output.keys():
    #                 if key in keys_to_iterate:
    #                     input_new[key] = output[key][i,...]
    #                 else:
    #                     input_new[key] = output[key]

    #             if i == 0:
    #                 output_new = self.recursive_output(input_new)
    #                 output_new = {key: torch.unsqueeze(output_new[key] if type(output_new[key]) == torch.Tensor else torch.tensor(output_new[key]), dim = 0) for key in output_new.keys()}
    #             else:
    #                 output_new_temp = self.recursive_output(input_new)
    #                 output_new_temp = {key: torch.unsqueeze(output_new_temp[key] if type(output_new_temp[key]) == torch.Tensor else torch.tensor(output_new_temp[key]), dim = 0) for key in output_new_temp.keys()}
    #                 output_new = {key: torch.cat([output_new[key], output_new_temp[key]], dim = 0) for key in output_new_temp.keys()}

    #         return output_new

    # def get_single_output_scipy(self, input_scipy):
    #     # make sure that the dim = 1
    #     # if the dim = 1 then if the structure is periodic then get the band gap and return the output
    #     # if the structure is periodic with defect then get the band gap then the defect peaks and return the output

    #     output = {}
    #     if type(input_scipy['R']) == PhysicalQuantity:
    #         R = input_scipy['R'].values.cpu().numpy()
    #         T = input_scipy['T'].values.cpu().numpy()
    #         wavelength = input_scipy['wavelength'].values.cpu().numpy()
    #         PBG_range_addition = input_scipy['PBG_range_addition'].values.cpu().numpy()
    #     elif type(input_scipy['R']) == torch.Tensor:
    #         R = input_scipy['R'].cpu().numpy()
    #         T = input_scipy['T'].cpu().numpy()
    #         wavelength = input_scipy['wavelength'].cpu().numpy()
    #         PBG_range_addition = input_scipy['PBG_range_addition'].cpu().numpy()
        
    #     output.update(self.try_get_PBG(wavelength, R, T, PBG_range=input_scipy['PBG_range'], PBG_range_addition=PBG_range_addition,close_enough=input_scipy['close_enough'], structure_type=input_scipy['strcuture_type'], number_of_defect_peaks=input_scipy['number_of_defect_peaks']))

    #     try:
    #         if input_scipy['strcuture_type'] == 'periodic_with_defect':
    #             output.update(
    #             self.defect_peak_analysis(
    #                 T,
    #                 output['band_gap_begining'], output['band_gap_end'],
    #                 number_of_defect_peaks=input_scipy['number_of_defect_peaks']
    #                 )
    #             )
    #     except Exception as e:
    #         print('The defect peak analysis failed with the following error: ', e)

    #     # convert all of the outputs to wavelength in form in torch.Tensor
    #     output['band_gap_begining'] = input_scipy['wavelength'][output['band_gap_begining']]
    #     output['band_gap_end']      = input_scipy['wavelength'][output['band_gap_end']]
    #     output['band_gap_width']    = output['band_gap_end'] - output['band_gap_begining']
    #     output['band_gap_center']   = (output['band_gap_begining'] + output['band_gap_end'])/2
    #     output['PBG_range']         = [output['band_gap_begining'], output['band_gap_end']] 
    #     output['PBG_range_addition'] = output['PBG_range_addition']
        
    #     if input_scipy['strcuture_type'] == 'periodic':
    #         return output
    #     try:
    #         # convert all of the outputs to wavelength in form in torch.Tensor
    #         defect_peak_width_in_indices = [output['index_right_defect'][i] - output['index_left_defect'][i] for i in range(len(output['index_right_defect']))]
    #         output['peak_intensity_R']  = [input_scipy['R'][output['peak_wavelength'][i]] for i in range(len(output['peak_wavelength']))]
    #         output['peak_intensity_T']  = [input_scipy['T'][output['peak_wavelength'][i]] for i in range(len(output['peak_wavelength']))]
    #         output['peak_index']        = [output['peak_wavelength'][i] for i in range(len(output['peak_wavelength']))]
    #         output['peak_wavelength']   = [input_scipy['wavelength'][output['peak_wavelength'][i]] for i in range(len(output['peak_wavelength']))]
    #         output['index_left_defect'] = [input_scipy['wavelength'][output['index_left_defect'][i]] for i in range(len(output['index_left_defect']))]
    #         output['index_right_defect'] = [input_scipy['wavelength'][output['index_right_defect'][i]] for i in range(len(output['index_right_defect']))]
    #         output['defect_peak_width'] = [output['index_right_defect'][i] - output['index_left_defect'][i] for i in range(len(output['index_right_defect']))]
    #         output['FullWidthHalfMax'] = [output['FullWidthHalfMax'][i] * output['defect_peak_width'][i] / defect_peak_width_in_indices[i] for i in range(len(output['index_right_defect']))]
    #     except Exception as e:
    #         print('The defect peak analysis failed with the following error: ', e)

    #     return output



    # def try_get_PBG(self,wavelength, R, T, PBG_range, PBG_range_addition, close_enough, structure_type: Literal['periodic', 'periodic_with_defect'] = 'periodic_with_defect', number_of_defect_peaks: int = 1):
    #     # First we get the band gap with the additional range
    #     # test if the number of defect peaks is equal to the number of peaks
    #     # and test if the new band gap close to the old band gap
    #     # if not add or subtract the PBG_range_addition and get the band gap again
    #     # iterate until the number of defect peaks is equal to the number of peaks and the new band gap is close to the old band gap
    #     # or the number of iteration is equal to 100 then return the band gap and the additional range
    #     max_iteration = 500
    #     iteration = 0
    #     correct_band_gap = False

    #     current_additional_range_1 = [PBG_range_addition[0], PBG_range_addition[1]]
    #     current_additional_range_2 = [PBG_range_addition[0], PBG_range_addition[1]]
    #     current_additional_range_3 = [PBG_range_addition[0], PBG_range_addition[1]]
    #     current_additional_range_4 = [PBG_range_addition[0], PBG_range_addition[1]]
    #     current_additional_range   = [PBG_range_addition[0], PBG_range_addition[1]]
    #     PBG_range_index = [np.argmin(np.abs(wavelength - PBG_range[0].item())), np.argmin(np.abs(wavelength - PBG_range[1].item()))]
    #     while not correct_band_gap and iteration < max_iteration:
    #         iteration += 1
    #         current_PBG_range = [PBG_range[0] - current_additional_range[0], PBG_range[1] + current_additional_range[1]]
    #         # convert the PBG from the wavelength to the index
    #         current_PBG_range = [np.argmin(np.abs(wavelength - current_PBG_range[0].item())), np.argmin(np.abs(wavelength - current_PBG_range[1].item()))]
    #         try:
    #             output = self.get_band_gap(R, structure_type=structure_type, number_of_defect_peaks=number_of_defect_peaks, PBG_range=current_PBG_range)
    #             peaks, peaks_properties = self.find_peaks(T[output['band_gap_begining']:output['band_gap_end']])
    #             correct_band_gap = (((abs(PBG_range[0] - wavelength[output['band_gap_begining']]) < close_enough) and (abs(PBG_range[1] - wavelength[output['band_gap_end']]) < close_enough))and (len(peaks) == number_of_defect_peaks))
    #         except Exception as e:
    #             pass

    #         if not correct_band_gap:
    #             if iteration%4 == 0:
    #                 current_additional_range = [current_additional_range_1[0] + PBG_range_addition[0]/10, #np.random.randint(-1,1)*np.random.rand(1)*PBG_range_addition[0],
    #                                             current_additional_range_1[1] + PBG_range_addition[1]/10 ] #np.random.randint(-1,1)*np.random.rand(1)*PBG_range_addition[1]]
    #                 current_additional_range_1 = current_additional_range
    #             elif iteration%4 == 1:
    #                 current_additional_range = [current_additional_range_2[0] - PBG_range_addition[0]/10, 
    #                                             current_additional_range_2[1] - PBG_range_addition[1]/10 ]
    #                 current_additional_range_2 = current_additional_range
    #             elif iteration%4 == 2:
    #                 current_additional_range = [current_additional_range_3[0] + PBG_range_addition[0]/10, 
    #                                             current_additional_range_3[1] - PBG_range_addition[1]/10 ]
    #                 current_additional_range_3 = current_additional_range
    #             elif iteration%4 == 3:
    #                 current_additional_range = [current_additional_range_4[0] - PBG_range_addition[0]/10, 
    #                                             current_additional_range_4[1] + PBG_range_addition[1]/10 ]
    #                 current_additional_range_4 = current_additional_range


    #     if not correct_band_gap:
    #         print('The band gap is not correct after {} iteration'.format(iteration))
    #     else:   
    #         output['PBG_range_addition'] = max(current_additional_range[0],close_enough), max(current_additional_range[1],close_enough)
    #         print('The band gap is correct after {} iteration'.format(iteration))
    #         print('The band gap is from {} to {}'.format(wavelength[output['band_gap_begining']], wavelength[output['band_gap_end']]))
    #         print('The additional range is from {} to {}'.format(output['PBG_range_addition'][0], output['PBG_range_addition'][1]))

    #     return output
