import torch
import numpy as np
import os
import random
import json

def get_random_problems(batch_size, problem_size, dist_type="uniform"):
    if dist_type == 'gaussian':
        problems = torch.randn(size=(batch_size, problem_size, 2))
    # problems.shape: (batch, problem, 2)
    elif dist_type == 'uniform': 
         problems = torch.rand(size=(batch_size, problem_size, 2))
    else:
        raise TypeError
    return problems


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




def normalize_coords(tensor):
    min_values = tensor.min(dim=0, keepdim=True)[0]
    max_values = tensor.max(dim=0, keepdim=True)[0]
    max_diff = (max_values - min_values).max()  # Choose the maximum span as the scaling factor

    normalized_tensor = (tensor - min_values) / max_diff
    return normalized_tensor


def extract_dimension(file_path):
    """Extract DIMENSION from a file, either from a .txt or .json file."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            for line in f:
                if "DIMENSION" in line:
                    return int(line.split()[-1])
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get("DIMENSION", None)
    else:
        raise ValueError("Unsupported file format: Only .txt and .json are supported.")
    
    return None  # Return None if no DIMENSION is found


def Shuffle_Filename_Loader(file_path, shuffle=True):
    """
    Return a dictionary where:
    - Keys are the shuffled txt file names.
    - Values are the extracted dimension values.
    """

    # Get all txt files
    txt_files = [f for f in os.listdir(file_path) if f.endswith('.txt')]

    # Shuffle file order
    if shuffle:
        random.shuffle(txt_files)

    # Create a dictionary {filename: dimension_value}
    shuffled_dict = {}
    for file_name in txt_files:
        file_path_full = os.path.join(file_path, file_name)
        dimension_value = extract_dimension(file_path_full)  # Extract the dimension value
        
        if dimension_value is not None:
            shuffled_dict[file_name] = dimension_value  # File name as the key

    return shuffled_dict

def extract_from_json(file_path):
    """Extract DIMENSION and NODE_COORDS from a single JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Extract dimension
    dimension = data.get("DIMENSION")
    
    # Extract coordinates and convert them to a tensor
    coords = data.get("NODE_COORDS", [])
    
    if not coords:
        return None  # If no coordinates are found, return None
    
    coord_tensor = normalize_coords(torch.tensor(coords, dtype=torch.float32))
    
    return (dimension, coord_tensor)

def extract_from_txt(file_path):
    """Process each txt file to extract DIMENSION and coordinate data"""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the DIMENSION value
    dimension = None
    for line in lines:
        if "DIMENSION" in line:
            dimension = int(line.split()[-1])  # Extract the number after DIMENSION
            break
    
    if dimension is None:
        return None  # If DIMENSION is not found, return None
    
    # Extract coordinate data between NODE_COORD_SECTION and EOF
    coords = []
    in_node_coord_section = False
    for line in lines:
        if "NODE_COORD_SECTION" in line:
            in_node_coord_section = True
            continue  # Skip the NODE_COORD_SECTION line
        if in_node_coord_section:
            if "EOF" in line:
                break  # Stop reading when EOF is found
            # Split each line by spaces, take the second and third numbers
            parts = line.split()
            if len(parts) >= 3:
                coords.append([float(parts[1]), float(parts[2])])
    
    # Convert coordinates to tensor and normalize
    coord_tensor = normalize_coords(torch.tensor(coords))
    
    # Return the result as a tuple (DIMENSION, coordinate tensor)
    return (dimension, coord_tensor)

# def extract_and_process_json(root_dir):
#     """
#     Extract DIMENSION and NODE_COORDS from all JSON files in the given directory.
#     Returns a dictionary where:
#     - Keys are the indices of JSON files (starting from 0).
#     - Values are tuples of (DIMENSION, coordinate tensor).
#     """
#     json_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.json')])
    
#     # Create a dictionary to store results
#     result_dict = {}
    
#     for idx, json_file in enumerate(json_files):
#         file_path = os.path.join(root_dir, json_file)
        
#         result = extract_from_json(file_path)
        
#         if result:
#             result_dict[idx] = result  # Store result in dictionary with index as the key
    
#     return result_dict

# def extract_and_process_txt(root_dir, problem_index):
#     # Get all txt files in the target directory
#     txt_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.txt')])
    
#     # Check if the problem_index is valid
#     if not isinstance(problem_index, int) or problem_index >= len(txt_files) or problem_index < 0:
#         raise IndexError("problem_index is out of range")
    
#     target_file = os.path.join(root_dir, txt_files[problem_index])
    
#     # Process the txt file
#     return process_txt_file(target_file)
   