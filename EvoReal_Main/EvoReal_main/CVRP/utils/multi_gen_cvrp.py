import numpy as np
import multiprocessing
import logging
import os
import glob
import re
import torch
import shutil
import time
import sys
import importlib

from utils.batch_gen_cvrp import batch_gen_cvrp
from llm_log.generate_instance_cvrp import batch_generate_cvrp_instances

def generate_and_solve_cvrp_batch(
    output_dir,
    n_total=10000,
    problem_size=100,
    gen_type=None,
    n_proc=20,
    logger=None,
    log_prefix=None,
):
    logger = get_console_logger("multi_gen_cvrp") if logger is None else logger
    os.makedirs(output_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    solved_counter = manager.Value('i', 0)
    max_valid = n_total

    part_files = []
    round_id = 1
    while solved_counter.value < max_valid:
        need = max_valid - solved_counter.value
        N_GENERATE = int(np.ceil(need * 1.5))  
        logger.info(f"[Round {round_id}] Generating {N_GENERATE} CVRP instances (need {need} valid).")
        all_instances = batch_generate_cvrp_instances(N_GENERATE, n=problem_size, log_prefix=log_prefix)
        batch_data = list(zip([ins['locations'] for ins in all_instances],
                              [ins['demands'] for ins in all_instances],
                              [ins['capacity'] for ins in all_instances]))

        N_PER_PROC = max(1, N_GENERATE // n_proc)
        jobs = []
        for i in range(n_proc):
            s, e = i * N_PER_PROC, (i + 1) * N_PER_PROC if i < n_proc - 1 else N_GENERATE
            if s >= e:
                break
            pfile = os.path.join(output_dir, f"type_{gen_type}{problem_size}_r{round_id}_{i+1:02d}.txt")
            part_files.append(pfile)
            p = multiprocessing.Process(target=batch_gen_cvrp,
                                        args=(batch_data[s:e], pfile, solved_counter, max_valid))
            p.start()
            jobs.append(p)

        for p in jobs:
            p.join()
        round_id += 1

    # merge the results
    merged_path = os.path.join(output_dir, f"type_{gen_type}{problem_size}_instances_formatted.txt")

    written = 0
    with open(merged_path, "w") as fout:
        for p in part_files:
            if os.path.exists(p):
                with open(p, "r") as fin:
                    for line in fin:
                        if written >= n_total:
                            break
                        fout.write(line)
                        written += 1
                os.remove(p)
            if written >= n_total:
                break
    logger.info(f"All results merged to: {merged_path}, total={written}")
    return merged_path

def setup_logger():
    logger = logging.getLogger("multi_gen_cvrp")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler("multi_gen_cvrp.log", mode="a")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    return logger


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

