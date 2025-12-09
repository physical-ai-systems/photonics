import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Dataset.Dataset import PhotonicDataset
from Dataset.MaterialDataset import MaterialDataset

def main():
    dataset = PhotonicDataset(
        structure_layers=20,
        ranges=(400, 700),
        steps=1,
        dataset_size=1000
    )
    
    samples = dataset[0]
    print("Sample keys:", samples.keys())

def main1():
    dataset = MaterialDataset(
        structure_layers=4,
        ranges=(400, 700),
        steps=1,
        dataset_size=1000)

if __name__ == "__main__":
    main1()