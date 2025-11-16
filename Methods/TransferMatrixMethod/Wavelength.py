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
