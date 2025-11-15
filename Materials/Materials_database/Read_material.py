import os

import scipy.interpolate
import torch
import yaml


class Read_material:
    """ Read the material from the yaml file and return the refractive index and the extinction coefficient """
    def __init__(self,
                 location,
                 name):
        """ 
        Args:
            location: The location of the material in the database
            name: The name of the material
        """
        self.filename = os.path.join(
            os.path.dirname(__file__),         
            'Materials',
            *location , name + '.yml'
            )
        self.name = name
        
        material = self.read_yaml()
        if 'REFERENCES' in material:
            self.reference = material['REFERENCES']
        if 'COMMENTS' in material:
            self.comment   = material['COMMENTS']

        if 'DATA' in material:
            for data in material['DATA']:
                
                if 'type' in data:
                    
                    data_type = data['type'].split()
                    if data_type[0] == 'tabulated':
                        self.data_type = data_type[0]
                        if 'variables' in data:
                            self.variables = data['variables'].split()
                        
                        if 'units' in data:
                            self.units = data['units']

                        if 'ranges' in data:
                            self.ranges = {}
                            ranges = data['ranges']
                            for variable in ranges:
                                string_ranges: str = ranges[variable]
                                self.ranges[variable] = [float(range) for range in string_ranges.split()]
                        
                        if 'data' in data:
                            new_data = data['data']
                            self.data = {}
                            for variable in new_data:

                                values = new_data[variable]
                                if values == 'None':
                                    self.data[variable] = None
                                else:
                                    self.data[variable] = torch.tensor([float(value) for value in values.split()])
                        if 'outputs' in data:
                            self.outputs = data['outputs'].split()

                    elif data_type[0] == 'formula':
                        self.data_type = data_type[0]
                        if len(data_type) > 1:
                            self.formula_num = int(data_type[1])

                        if 'coefficients' in data:
                            coefficients = data['coefficients'].split()
                            self.coefficients = [float(coefficient) for coefficient in coefficients]

                        if 'variables'in data:
                            self.variables = data['variables'].split()

                        if 'units' in data:
                            self.units = data['units']

                        if 'ranges' in data:
                            self.ranges = {}
                            ranges = data['ranges']
                            for variable in ranges:
                                string_ranges: str = ranges[variable]
                                self.ranges[variable] = [float(r) for r in string_ranges.split()]

                        if 'outputs' in data:
                            self.outputs = data['outputs'].split()

                    self.array_variables = ['wavelength']



    def read_yaml(self):
        """ Read the yaml file"""

        f = open(self.filename)
        try:
            material = yaml.safe_load(f)
        except yaml.YAMLError:
            raise Exception('Bad Material YAML File.')
        finally:
            f.close()

        return material


    def get_refractiveindex(self, list_of_variables, list_of_values):

        """
        Get the refractive index at a certain wavelenght and values of the variables
        Args:
            wavelength: The wavelength in meters
        """

        # Prepare the variables and values    
        variables, values = self.prepare_variables_and_values(list_of_variables, list_of_values)


        if self.data_type == 'tabulated':
            outputs = self.get_variable_value(variables, values)

            # # Interpolate the data
            # n = scipy.interpolate.griddata(self.data[variables[0]], self.data[variables[1]], values, method='linear')

        elif self.data_type == 'formula':    
            # Calculate the refractive index
            outputs = self.formula(self.formula_num, self.coefficients, values)

        if len(self.outputs) == 1 and self.outputs[0] == 'n':
            n = outputs[0]
        elif len(self.outputs) == 2 and self.outputs[0] == 'n' and self.outputs[1] == 'k':
            n = outputs[0] + 1j * outputs[1]
        else:
            raise Exception('The outputs are not correct')
        return n


    def prepare_variables_and_values(self, list_of_variables, list_of_values):
        """ Prepare the variables and values for the formula """

        # First check if the variables and values are correct
        assert len(list_of_variables) == len(list_of_values), "The number of variables should be equal to the number of values"

        for value, variable in zip(list_of_values, list_of_variables):
            assert variable in self.variables, f"The variable {variable} is not defined in the material {self.name}"
            
            if variable not in self.array_variables:
                assert max(value) <= self.ranges[variable][1] and min(value) >= self.ranges[variable][0], f"The value of {variable} is out of range"
            
            else:
                assert type(value) is torch.Tensor,  "The value of " + variable + " should be a torch tensor"
                assert torch.all(value <= self.ranges[variable][1]) and torch.all(value >= self.ranges[variable][0]), "The value of " + variable + " is out of range"

        # Then prepare the variables and values
        values    = []
        for variable in self.variables:
            if variable in list_of_variables:
                values.append(list_of_values[list_of_variables.index(variable)])
            else:
                values.append((self.ranges[variable][1])/2)

        return self.variables, values


    def formula(self, formula_num, coefficients, values):
        """ This function is to calculate variable/s from the coefficients and the values"""
        variable = []
        if formula_num == 1:
            variable.append(coefficients[0] * values[0] + coefficients[1] * values[1] + coefficients[2] * values[2] + coefficients[3] * values[2]**2 + coefficients[4])
        else :  
            raise NotImplementedError('Formula not implemented yet')
        return variable
    
    def get_variable_value(self, variables, values):
        """ Make a grid of the variables """
        if len(variables) > 2:
            raise NotImplementedError('The number of variables should be less than 3')
            #TODO: get all indices of the variables and then iterate over them to get the value of the variable
        
        # get the index the output
        for variable, value in zip(variables, values):
            # find the variable in the list of variables
            assert variable in self.variables, f"The variable {variable} is not defined in the material {self.name}"
            # get the index of self.data[variable]
            if self.data[variable] is None:
                continue
            value = torch.tensor(value)  # Convert to a PyTorch tensor if it's not already
            indices = [torch.where(self.data[variable] == val)[0] for val in value]
        # get the value of the variable
        outputs = []
        for output in self.outputs:
            assert output in self.data, f"The output {output} is not defined in the material {self.name}"
            for index in indices:
                outputs.append(self.data[output][index])

        # convert the variable to torch tensor
        for i,output in enumerate(outputs):
            if output is not None and not torch.is_tensor(output):
                outputs[i] = torch.tensor([output])


        # check if the variable shape is equal wavelength shape
        for i,output in enumerate(outputs):
            if output is not None:
                if len(output) != len(values[variables.index('wavelength')]):
                    if len(output) == 1:
                        #TODO: replace this with broadcasting torch function
                        outputs[i] = output * torch.ones(values[variables.index('wavelength')].shape)
                    else:
                        raise Exception(f"The shape of the variable {output} is not equal to the shape of the wavelength {values[0]}")
        outputs = [torch.stack(outputs, dim=0)]
        return outputs