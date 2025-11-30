import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Dataset.MaterialDataset import SqliteMaterialDataset

def test_dataset():
    print("Initializing SqliteMaterialDataset...")
    try:
        dataset = SqliteMaterialDataset(
            ranges=(400, 700), 
            steps=1, # Changed from step_size=0.1 to match new init
            batch_size=2,
            num_layers=5
        )
    except Exception as e:
        print(f"Failed to init dataset: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"Dataset created with {len(dataset)} materials.")
    
    if len(dataset) == 0:
        print("Warning: No materials found.")
        return

    print("Testing first sample:")
    sample = dataset[0]
    
    print(f"Keys: {sample.keys()}")
    if 'name' in sample:
        print(f"Name: {sample['name']}")
    
    if 'R' in sample:
        print(f"R shape: {sample['R'].shape}")
        print(f"T shape: {sample['T'].shape}")
        print(f"Material Choice shape: {sample['material_choice'].shape}")
        print(f"Thickness shape: {sample['layer_thickness'].shape}")
        
        # Check values
        print(f"R (first batch, first point): {sample['R'][0,0]}")
        print(f"T (first batch, first point): {sample['T'][0,0]}")

if __name__ == "__main__":
    test_dataset()