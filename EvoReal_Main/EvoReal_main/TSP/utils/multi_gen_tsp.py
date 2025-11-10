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
from utils.batch_gen_tsp import batch_gen_tsp


def get_console_logger(name=None):
    logger = logging.getLogger(name or f"console_logger_{os.getpid()}")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    # avoid repeatedly adding handler
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

def load_generator(gen_type: str = None, log_prefix=""):
    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent.parent  # Point to the root directory of TSP
    if gen_type is None:
        raise NotImplementedError("Please specify the generation type!")

    module_map = {
        "S1": ("gpt_S1", "generate_s1_type"),
        "S2": ("gpt_S2", "generate_s2_type"),
        "S3": ("gpt_S3", "generate_s3_type")
    }



    module_file, func_name = module_map[gen_type]
    module_path = os.path.join(BASE_DIR, f"{log_prefix}_llm_log", f"{module_file}.py")
    
    # dynamic loading modules
    spec = importlib.util.spec_from_file_location(module_file, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    generator = getattr(module, func_name)
    return generator

def setup_logger():
    logger = logging.getLogger(f"multi_gen_tsp_type_{gen_type}")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"multi_gen_tsp_type_{gen_type}.log", mode="a")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    fh.setFormatter(formatter)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(fh)
    return logger


def process_and_merge_tours(output_dir, gen_type, problem_size=100):
    #Use an asterisk (*) to cover all instance files
    pattern = os.path.join(output_dir, f"tsp{problem_size}_instance*_type{gen_type}.txt")
    file_list = sorted(glob.glob(pattern))
    # Use regular expressions to further ensure format matching
    regex = re.compile(rf"tsp{problem_size}_instance\d+_type{gen_type}\.txt$")
    file_list = [f for f in file_list if regex.search(os.path.basename(f))]
    if not file_list:
        print("No matching txt file found!")
        return

    merged_file = os.path.join(output_dir, f"tsp{problem_size}_type{gen_type}_merged.txt")
    with open(merged_file, 'w', encoding='utf-8') as fout:
        for fname in file_list:
            with open(fname, encoding='utf-8') as fin:
                for line in fin:
                    if "output" in line:
                        coord, tourstr = line.strip().split("output", 1)
                        tour = tourstr.strip().split()[0:problem_size]
                        tour = [str(int(x)) for x in tour]
                        tour.append("1")
                        fout.write(f"{coord.strip()} output {' '.join(tour)}\n")
    print(f"Process done. Merged file has been written to: {merged_file}")

    # Delete the original shard file
    for fname in file_list:
        os.remove(fname)
    print("All original sliced data files has been deleted.")

def prepare_all_data(n_total, problem_size, logger, gen_type = None, generator = None):
    if generator is None or gen_type is None:
        raise NotImplementedError
    batch = generator(batch_size=n_total, problem_size=problem_size)
    np.save(f"all_generated_tsp_type{gen_type}.npy", batch.numpy())
    if logger is not None:
        logger.info(f"Generated {n_total} TSP {gen_type}-type problems.")
    return batch  # Return for subsequent slicing of the main process

def clean_concorde_temp_files():
    patterns = [
        "problem.sol", "problem.sav", "problem.res", "problem.pul", "problem.mas",
        "Oproblem.*", "Nproblem.*", "problem.tsp", "problem.tour",
        "*.sol", "*.sav", "*.res", "*.pul", "*.mas", "*.sav~", "*.mas~",
        "*.npy"
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
            except Exception as e:
                pass
            
def detect_concorde():
    path = os.environ.get("CONCORDE_PATH")
    if path and os.path.exists(path):
        return path
    if shutil.which("concorde") is not None:
        return "concorde"
    default_path = ""          # your path to concorde.exe
    if os.path.exists(default_path):
        return default_path
    raise FileNotFoundError("Concorde not found.")


def generate_and_solve_tsp_batch(
    output_dir,
    n_total=10000,
    problem_size=100,
    gen_type="C",
    n_proc=5,
    logger=None,
    log_prefix = ""
):
    """
    High-level entry: Batch generation, concurrent solution, merge and format TSP data, and return the merged txt path
    """
    N_TOTAL = int(n_total*1.5) # allow to generate a bit more to ensure data amount equality
    if logger is not None:
        logger = setup_logger()
    else:
        logger = get_console_logger('batch_gen_tsp')
    os.makedirs(output_dir, exist_ok=True)
    generator = load_generator(gen_type=gen_type,log_prefix=log_prefix)
    batch = prepare_all_data(n_total=N_TOTAL, problem_size=problem_size, logger=logger, gen_type=gen_type, generator=generator)
    np.save(os.path.join(output_dir, f"all_generated_tsp_type{gen_type}.npy"), batch.numpy())

    all_coords = [batch[i].numpy() for i in range(N_TOTAL)]

    from utils.batch_gen_tsp import detect_concorde, batch_gen_tsp
    concorde_path = detect_concorde()
    manager = multiprocessing.Manager()
    solved_counter = manager.Value('i', 0)
    max_valid = n_total

    N_PER_PROC = n_total // n_proc
    jobs = []
    for idx in range(n_proc):
        start = idx * N_PER_PROC
        end = (idx+1) * N_PER_PROC if idx<n_proc-1 else N_TOTAL
        save_file = os.path.join(output_dir, f"tsp{problem_size}_instance{idx+1:02d}_type{gen_type}.txt")
        batch_coords = [torch.tensor(c) for c in all_coords[start: end]]
        p = multiprocessing.Process(
            target=batch_gen_tsp,
            args=(batch_coords, save_file, concorde_path, logger, solved_counter, max_valid)
        )
        p.start()
        jobs.append(p)

    while True:
        if solved_counter.value >= max_valid:
            for p in jobs:
                if p.is_alive():
                    p.terminate()
            break
        if all([not p.is_alive() for p in jobs]):
            break
        time.sleep(2)
    
    # MERGE
    process_and_merge_tours(output_dir, gen_type, problem_size=problem_size)
    merged_file = os.path.join(output_dir, f"tsp{problem_size}_type{gen_type}_merged.txt")
    clean_concorde_temp_files()
    return merged_file


gen_type = 'C'
   
if __name__ == "__main__":
    logger = setup_logger()
    N_TOTAL = 15000
    N_PROC = 5
    N_PER_PROC = 5000
    PROBLEM_SIZE = 100
    concorde_path = detect_concorde()
    generator = load_generator(gen_type=gen_type)
    logger.info("Start generating all data...")
    
    batch = prepare_all_data(N_TOTAL, PROBLEM_SIZE, logger, gen_type=gen_type, generator=generator)
    # batch= (N_TOTAL, 100, 2) torch tensor
    all_coords = [batch[i].numpy() for i in range(N_TOTAL)]  # Converting it to a list is convenient for slicing and multi-process passing

    logger.info("Start multiprocessing...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, f"result_type{gen_type}")
    os.makedirs(output_dir, exist_ok=True)

    manager = multiprocessing.Manager()
    solved_counter = manager.Value('i', 0)
    max_valid = 10000

    jobs = []
    for idx in range(N_PROC):
        start = idx * N_PER_PROC
        save_file = os.path.join(output_dir, f"tsp100_instance{idx+1:02d}_type{gen_type}.txt")
        batch_coords = [torch.tensor(c) for c in all_coords[start: start + N_PER_PROC]]  # PASS TO THE SUBPROCESS OF batch
        p = multiprocessing.Process(
            target=batch_gen_tsp,
            args=(batch_coords, save_file, concorde_path, logger, solved_counter, max_valid)
        )
        p.start()
        jobs.append(p)
        

    # --- Monitor the count. kill all if the limit is exceeded ---
    import time
    while True:
        if solved_counter.value >= max_valid:
            logger.info(f"Total valid solutions reached {max_valid}, terminating all subprocesses.")
            for p in jobs:
                if p.is_alive():
                    p.terminate()
            break
        if all([not p.is_alive() for p in jobs]):
            break
        time.sleep(3)
    logger.info("All batches finished.")
    clean_concorde_temp_files()
    logger.info("All Concorde temp files have been deleted.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(BASE_DIR, f"result_type{gen_type}")
    process_and_merge_tours(output_dir, gen_type, problem_size=100)