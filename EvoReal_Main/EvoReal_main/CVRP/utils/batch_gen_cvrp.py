import hygese as hgs
import os
import numpy as np
import subprocess
import tempfile
import logging
import shutil
import sys

def get_console_logger(name=None):
    logger = logging.getLogger(name or f"console_logger_{os.getpid()}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger


def _build_record(locations, demands, capacity, routes, x, y):
    try:
        if locations is None or demands is None or capacity is None or routes is None:
            return None
        if locations.ndim != 2 or locations.shape[1] != 2:
            return None
        if not np.isfinite(locations).all():
            return None

        V = locations.shape[0] - 1
        if V <= 0:
            return None

        # demand & capacity
        if len(demands) != V + 1:
            return None
        if not np.isfinite(demands).all():
            return None
        if int(demands[0]) != 0:
            return None

        cap = int(capacity)
        if cap <= 0:
            return None

        # routes have to cover all customers
        visited = [int(n) for r in routes for n in r]
        if sorted(visited) != list(range(1, V + 1)):
            return None

        # tour + flags
        tour, flags = [], []
        for r in routes:
            if not r:
                return None
            for j, node in enumerate(r):
                tour.append(int(node))
                flags.append(1 if j == 0 else 0)
        if len(tour) != V or len(flags) != V:
            return None

        # cost
        cost_val = _euclidean_cost_from_routes(routes, x, y)
        if not np.isfinite(cost_val) or cost_val <= 0:
            return None

        # Concatenate into a single string
        parts = []
        parts.append(f"depot,{x[0]:.3f},{y[0]:.3f},")
        parts.append(f"customer,{x[1]:.3f},{y[1]:.3f},")
        for i in range(2, len(x)):
            parts.append(f"{x[i]:.3f},{y[i]:.3f},")
        parts.append(f"capacity,{cap},")
        parts.append("demand," + ",".join(str(int(d)) for d in demands) + ",")
        parts.append(f"cost,{cost_val:.3f},")
        parts.append("node_flag," + ",".join(str(v) for v in (tour + flags)) + "\n")
        return "".join(parts)
    except Exception:
        return None


def batch_gen_cvrp(instances, save_path, solved_counter=None, max_valid=None, logger=None):
    if logger is None:
        logger = get_console_logger("batch_gen_cvrp")

    with open(save_path, "w") as f:
        for idx, (locations, demands, capacity) in enumerate(instances):
            try:
                if solved_counter is not None and max_valid is not None and solved_counter.value >= max_valid:
                    logger.info(f"Process {os.getpid()}: Early exit, global valid={max_valid}")
                    break

                x, y = locations[:, 0], locations[:, 1]

                ap = hgs.AlgorithmParameters(
                    timeLimit=2, nbGranular=20, mu=25, lambda_=40,
                    nbElite=4, nbIter=25000, targetFeasible=0.2
                )
                solver = hgs.Solver(parameters=ap, verbose=False)
                result = solver.solve_cvrp({
                    "x_coordinates": x, "y_coordinates": y, "demands": demands,
                    "vehicle_capacity": capacity,
                    "num_vehicles": int(np.ceil(sum(demands[1:]) / capacity))+1,
                    "service_times": np.zeros(len(demands)), "depot": 0
                })

                record = _build_record(locations, demands, capacity, result.routes, x, y)
                if record is None:
                    logger.warning(f"Problem {idx}: invalid (missing/incorrect depot/customers/capacity/demand/node_flag). Skipped.")
                    continue

                f.write(record)
                if solved_counter is not None and max_valid is not None:
                    solved_counter.value += 1
                    if solved_counter.value % 100 == 0:
                        logger.info(f"[{os.getpid()}] progress: global_valid={solved_counter.value}")

            except Exception as e:
                logger.error(f"Problem {idx}: Exception {e}. Skipped.")
                continue
            
            
            
def _euclidean_cost_from_routes(routes, x, y):
    """Calculate the total distance using routes + coordinates; Make sure that the beginning and end of each route are connected to depot(0)."""
    total = 0.0
    for r in routes:
        if not r:
            continue
        seq = list(r)
        # Make sure the beginning and the end are the depot 0
        if seq[0] != 0:
            seq = [0] + seq
        if seq[-1] != 0:
            seq = seq + [0]
        # get Euclidean distance
        for a, b in zip(seq, seq[1:]):
            total += float(np.hypot(x[a] - x[b], y[a] - y[b]))
    return total
