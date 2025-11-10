import numpy as np
import random
import torch
######################################################### Instance Generation Part ###########################################################

# trivial seed cvrp generator
def generate_customer_positions(n, cust_type=None):
    """
    Generate customer positions in [0,1] x [0,1] with three types: R, C, RC.
    R: Random uniform; C: Clustered; RC: Half clustered, half random.
    """
    CUST_TYPES = ['P1']
    if cust_type is None:
        cust_type = random.choice(CUST_TYPES)
    if cust_type == 'P1':
        # Uniformly sample n points, round to 3 decimals
        points = np.round(np.random.rand(n, 2), 3)
        return ensure_n_unique_points(points, n),cust_type

    else:
        raise ValueError(f"Unknown customer type: {cust_type}")

def generate_demands(n, demand_type=None, customer_positions=None):
    
    DEMAND_TYPES = ['Q1']
    if demand_type is None:
        demand_type = random.choice(DEMAND_TYPES)
    if demand_type == 'Q1': # Uniform [1,10]
        return np.random.randint(1, 11, size=n)

    else:
        raise ValueError(f"Unknown demand type: {demand_type}")
    


###################################################################################################################################
##################################################################################################################################
# non-evovlable components
def generate_depot(depot_type):
    """
    Generate depot position in [0,1] x [0,1].
    S1: Center; S3: Random; S2: Fixed origin.
    """
    if depot_type == 'S1':  # Center
        return np.array([0.5, 0.5])
    elif depot_type == 'S2':  # Eccentric (random in [0,1])
        return np.round(np.random.rand(2), 3)
    elif depot_type == 'S3':  # Fixed at origin
        return np.array([0.0, 0.0])
    else:
        raise ValueError(f"Unknown depot type: {depot_type}")

def generate_capacity(demands, r=None, k=2):
    """
    Calculate vehicle capacity based on demand distribution.
    demands: np.array, with demands[0]=0 for depot, demands[1:] for customers.
    r: controlling parameter, sampled if None.
    """
    n = len(demands) - 1  # Number of customers
    if r is None:
        r = np.random.triangular(3, 6, 25)
    customer_demands = demands[1:]
    avg_part = r * customer_demands.sum() / n
    max_part = k * customer_demands.max()
    capacity = max(np.ceil(avg_part), np.ceil(max_part))
    return capacity

def generate_single_instance(
    depot_type, customer_type=None, demand_type=None, n: int = 100
):
    depot = generate_depot(depot_type)
    cust_pos,cus_tag = generate_customer_positions(n, customer_type)
    demands,demand_tag = generate_demands(n, demand_type, cust_pos)
    capacity = generate_capacity(demands=demands, k=2)
    
    locations = np.vstack([depot, cust_pos])
    demands = np.insert(demands, 0, 0)
    return dict(
        n=n,
        depot=depot,
        customer_positions=cust_pos,
        demands=demands,
        capacity=capacity,
        locations=locations,
        customer_type = cus_tag,
        demand_type = demand_tag
    )

######################################################### Dataset Generation Part ###########################################################

def batch_generate_cvrp_instances(num_instances, n: int = 100):
    # Mixed, anonymized type tags for depot, position, demand
    DEPOT_TYPES = ['S1', 'S2', 'S3']

    instances = []
    for _ in range(num_instances):
        depot_type = random.choice(DEPOT_TYPES)

        instance = generate_single_instance(depot_type=depot_type, customer_type=None, demand_type=None, n=n)
        instance['depot_type'] = depot_type

        instance = normalize_cvrp_instance(instance)
        instances.append(instance)
    return instances



def to_pomo_input(instances):
    """
    Convert a list of instances to batched PyTorch tensors for model input.
    Each instance contains:
        locations: (n+1, 2), where the first row is the depot
        demands: (n+1, ), where the first element is the depot demand
    Returns:
        batch_size: int
        problem_size: int
        depot_xy: Tensor of shape (batch, 1, 2)
        node_xy: Tensor of shape (batch, problem, 2)
        node_demand: Tensor of shape (batch, problem)
    """
    batch_size = len(instances)
    problem_size = instances[0]['n']
    depot_xy = []
    node_xy = []
    node_demand = []
    for ins in instances:
        depot_xy.append(ins['depot'])
        node_xy.append(ins['customer_positions'])
        node_demand.append(ins['demands'][1:])
    depot_xy = torch.tensor(depot_xy).float().unsqueeze(1)      # (batch, 1, 2)
    node_xy = torch.tensor(node_xy).float()                     # (batch, problem, 2)
    node_demand = torch.tensor(node_demand).float()             # (batch, problem)
    return batch_size, problem_size, depot_xy, node_xy, node_demand

######################################################### utils ###########################################################
def normalize_cvrp_instance(instance):
    """
    Only normalize demand and capacity, not positions.
    """
    cap = instance['capacity']
    instance['demands'] = instance['demands'] / cap
    if not np.all(instance['demands'][1:] <= 1 + 1e-8):
        print("[Warning] Some demands exceed capacity after normalization, will be clipped.")
        instance['demands'] = np.clip(instance['demands'], 0, 1)
    instance['capacity'] = 1.0
    return instance

def ensure_n_unique_points(points, n):
    """
    Ensure the output points are unique and exactly n.
    """
    points = np.unique(points, axis=0)
    while len(points) < n:
        new_pt = np.round(np.random.rand(1, 2), 3)
        # Avoid duplicates
        while any((new_pt == points).all(axis=1)):
            new_pt = np.round(np.random.rand(1, 2), 3)
        points = np.vstack([points, new_pt])
    if len(points) > n:
        points = points[:n]
    return points
