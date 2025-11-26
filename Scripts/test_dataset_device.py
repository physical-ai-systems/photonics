import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from Dataset.Dataset import PhotonicDataset

def test_device(device_name):
    print(f"Testing device: {device_name}")
    try:
        device = torch.device(device_name)
        dataset = PhotonicDataset(
            num_layers=5,
            ranges=(400, 700),
            steps=100,
            dataset_size=10,
            device=device
        )
        
        # Check if wavelength is on device
        if dataset.wavelength.values.device.type != device.type:
            print(f"Error: Dataset wavelength is on {dataset.wavelength.values.device}, expected {device}")
            return False

        # Check __getitem__
        sample = dataset[0]
        
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if value.device.type != device.type:
                    print(f"Error: Output {key} is on {value.device}, expected {device}")
                    return False
        
        print(f"Success for device: {device_name}")
        return True
    except Exception as e:
        print(f"Exception testing {device_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Test CPU
    if not test_device('cpu'):
        sys.exit(1)
    
    # Test CUDA if available
    if torch.cuda.is_available():
        if not test_device('cuda'):
            sys.exit(1)
    else:
        print("CUDA not available, skipping CUDA test.")

if __name__ == "__main__":
    main()
