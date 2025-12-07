import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from Dataset.Dataset import PhotonicDataset
import matplotlib.pyplot as plt


def main():
    dataset = PhotonicDataset(
        structure_layers=20,
        ranges=(400, 700),
        steps=1,
        dataset_size=1000
    )
    
    samples = dataset[0]
    print("Sample keys:", samples.keys())

if __name__ == "__main__":
    main()