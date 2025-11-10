import os
import re
import torch
import json
import numpy as np
import logging
from pathlib import Path

def normalize_coords(array):
    """
    Normalize TSP coordinate data to the range [0,1] while maintaining proportions.
    Return a numpy.array type set of coordinates
    """
    min_values = np.min(array, axis=0, keepdims=True)
    max_values = np.max(array, axis=0, keepdims=True)
    max_diff = np.max(max_values - min_values)  # Choose the largest span as the scaling factor

    normalized_array = (array - min_values) / max_diff
    return normalized_array

def augment_xy_data_by_8_fold(problems, batch_size=1):
    # problems.shape: (batch, problem, 2)

    # Extract x and y as 2D tensors
    x = problems[:, 0].unsqueeze(1)  # (batch, problem, 1)
    y = problems[:, 1].unsqueeze(1)  # (batch, problem, 1)
    x = x.unsqueeze(0).expand(batch_size,-1,-1)
    y = y.unsqueeze(0).expand(batch_size,-1,-1)
    # Create 8 different transformed data sets
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    # Concatenate the 8 transformed data sets
    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    
    # Return the augmented data
    return aug_problems


def load_labels(root_dir=None):
    if root_dir is None:
        root_dir = Path(__file__).resolve().parent
    label_path = Path(root_dir) / "utils" / "labels_with_avg_opt.json"
    with open(label_path, 'r') as f:
        label_dict = json.load(f)
    return label_dict



def process_txt_file(file_path):
    """Extract DIMENSION and coordinates from a file, and return the normalized tensor."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Extract DIMENSION
    dimension = None
    for line in lines:
        if "DIMENSION" in line:
            dimension = int(line.split()[-1])
            break
    
    if dimension is None:
        return None
    
     # Extract coordinates: skip non-numeric lines, split by space or tab, keep only two float columns
    coords = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            try:
                x, y = float(parts[-2]), float(parts[-1])
                coords.append([x, y])
            except ValueError:
                continue  # skip non-numeric lines

    if len(coords) == 0:
        return None

    coord_tensor = torch.tensor(normalize_coords(coords),dtype=torch.float32)
    return (dimension, coord_tensor)


def process_json_file(file_path):
    """Extract DIMENSION and coordinates from a JSON file, and return the normalized tensor."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return None

    dimension = data.get("DIMENSION")
    coords = data.get("NODE_COORDS")

    if not dimension or not coords or len(coords) != dimension:
        return None

    coord_tensor = torch.tensor(normalize_coords(coords),dtype=torch.float32)
    return (dimension, coord_tensor)



