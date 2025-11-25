import torch
import torch.nn as nn


class PhysicalQuantity(nn.Module):
    def __init__(self, values=None, units=None, unit_prefix=None, ranges=None, steps=None, name=None, reference_index=None, reference_dimension=None, scale=None, shift_from_center=2, **kwargs):
        super().__init__()
        assert values is None or (ranges is None and steps is None), 'Either values or ranges and steps must be provided'
        self.values = torch.as_tensor(values) if values is not None else torch.arange(ranges[0], ranges[1]+steps, steps)
        self.units, self.unit_prefix, self.prefix_ineffective = units, unit_prefix, False
        self.steps  = steps   
        self.name   = name
        self.scale  = scale if scale is not None else self.get_scale_from_prefix(unit_prefix) if unit_prefix is not None else 1
        self.scale_values()
        self.shift_from_center = shift_from_center



    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        self._values = values
    
    @property
    def ranges(self):
        return [self.values.min(), self.values.max()]
    
    def to(self, device):
        self.values = self.values.to(device)
        return self
    @property
    def device(self):
        return self.values.device
    
    def scale_values(self, scale=None):
        if scale == None:
            scale = self.scale
        
        if self.prefix_ineffective:
            print('Warning: Prefix is already scaled, scaling again may lead to incorrect results.')
        
        self.values = self.values * scale
        if self.steps is not None:
            self.steps  = self.steps  * scale
        self.prefix_ineffective = True        # already scaled
        
    @property
    def center(self):
        return (self.ranges[0] + self.ranges[1])/(self.shift_from_center if self.shift_from_center is not None else 2)

    
    def broadcast(self, shape):
        self.values = torch.broadcast_to(self.values, shape)

    def get_in_unit(self, output_prefix=None):
        input_prefix = ''
        if output_prefix is None:
            output_prefix = self.unit_prefix
        return self.values * self.unit_conversion(input_prefix, output_prefix)
    
    def set_values_in_unit(self, output_prefix=None):
        self.values = self.get_in_unit(self.unit_prefix)
        self.units  = self.get_unit(self.unit_prefix)
        self.unit_prefix = None
        self.scale       = 1
    
    def unit_conversion(self, input_prefix, output_prefix):
        if input_prefix == output_prefix:
            return 1
        input_scale = self.get_scale_from_prefix(input_prefix)
        output_scale = self.get_scale_from_prefix(output_prefix)
        return input_scale / output_scale

    def get_scale_from_prefix(self, unit_prefix):
        scale_factors = [1e21, 1e18, 1e15, 1e12, 1e9, 1e6, 1e3, 1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21]
        unit_prefixs  = [ 'Z',  'E',  'P',  'T', 'G', 'M', 'k', '', 'm',  'u',  'n',   'p',   'f',   'a',   'z']
        return scale_factors[unit_prefixs.index(unit_prefix)] if unit_prefix is not None else 1
    
    def get_unit(self, prefix=None):
        if prefix is None:
            return self.units
        else:
            return prefix + self.units


    def __str__(self):
        return f"{self.name} with ranges {self.ranges} and steps {self.steps} in {self.unit_prefix if not self.prefix_ineffective else '' +self.units} units"
    
    def __repr__(self):
        return f"{self.name} with ranges {self.ranges} and steps {self.steps} in {self.unit_prefix if not self.prefix_ineffective else '' +self.units} units"
    
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __iter__(self):
        return iter(self.values)
    
    def __next__(self):
        return next(self.values)
    
    def __add__(self, other):
        return self.values + other
    
    def __sub__(self, other):
        return self.values - other
    
    def __mul__(self, other):
        return self.values * other
    
    def __truediv__(self, other):
        return self.values / other
    
    def __floordiv__(self, other):
        return self.values // other
    
    def __mod__(self, other):
        return self.values % other
    
    def __pow__(self, other):
        return self.values ** other
    
    def __lshift__(self, other):
        return self.values << other
    
    def __rshift__(self, other):
        return self.values >> other
    
    def __and__(self, other):
        return self.values & other
    
    def __xor__(self, other):
        return self.values ^ other
    
    def __or__(self, other):
        return self.values | other
    
    def __lt__(self, other):
        return self.values < other
    
    def __le__(self, other):
        return self.values <= other
    
    def __eq__(self, other):
        return self.values == other
    
    def __ne__(self, other):
        return self.values != other
    
    def __gt__(self, other):
        return self.values > other
    
    def __ge__(self, other):
        return self.values >= other
    
    def __neg__(self):
        return -self.values
    
    def __pos__(self):
        return +self.values
    
    def __abs__(self):
        return abs(self.values)
    
    
    @property
    def shape(self):
        return self.values.shape