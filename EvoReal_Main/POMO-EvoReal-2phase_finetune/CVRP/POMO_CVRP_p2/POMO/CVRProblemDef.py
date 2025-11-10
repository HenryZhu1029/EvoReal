
import torch
import numpy as np
import os
import re
import random

def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data



def Shuffle_PTFilename_Loader(pt_dir, shuffle=True):
    """
    Traverse all .pt files in the pt_dir directory in random order (optional).
    Return a dictionary: {pt file path: num_of_customers}
    num_of_customers is extracted using a regular expression from the file name, e.g., from X-n101-k25.pt, extract 101-1=100.
    """

    pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]

    if shuffle:
        random.shuffle(pt_files)

    train_dict = {}
    for fname in pt_files:
        # use regular expression to extract num_of_customers from filename
        match = re.search(r'-n(\d+)-k', fname)
        if match:
            num_of_customers = int(match.group(1)) - 1
            # file_path_full = os.path.join(pt_dir, fname)
            train_dict[fname] = num_of_customers
        else:
            print(f"Warning: {fname} does not match pattern, skipped.")

    return train_dict
