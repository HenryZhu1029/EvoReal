import os
import sys
import importlib
import json
import shutil
from pathlib import Path
import numpy as np
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from utils import multi_gen_cvrp
from LEHD.train_longTrain import main_train


def get_fitness_score(
    checkpoint_folder,
    idx_iteration,
    idx_response_id,
    cuda_device_num,
    module_type="cvrp",
    n_total=10000,
    problem_size=100, # fixed problem size
    n_proc=20,
    log_prefix = None,
    obj_type = 'min'
):

    module_type_upper = module_type.upper()


    gen_type = module_type_upper
    
    checkpoint_save_folder = os.path.join(checkpoint_folder,f"iter_{idx_iteration}_resp_{idx_response_id}")
    os.makedirs(checkpoint_save_folder, exist_ok=True)
    log_path = log_prefix + "_llm_log"
    # if idx_iteration !=0 or idx_response_id !=0:

    data_dir = os.path.join(ROOT_DIR, "llm_log", log_path, "gpt","data", f"iter_{idx_iteration}_resp_{idx_response_id}")
    os.makedirs(data_dir, exist_ok=True)

    if idx_iteration == 0 and idx_response_id == 0:
        log_prefix = None
        # data_dir = os.path.join(ROOT_DIR, "LEHD", "data")
        # merged_txt = os.path.join(data_dir, f"type_{module_type_upper}{problem_size}_instances_formatted.txt")
    # === 1. batch generation and solve ===

    # else:
    merged_txt = multi_gen_cvrp.generate_and_solve_cvrp_batch(
        output_dir=data_dir,
        n_total=n_total,
        problem_size=problem_size,
        gen_type=gen_type,
        n_proc=n_proc,
        logger= None,  # set to none for outputing logger to console instead of saving logs
        log_prefix = log_prefix
    )



    fitness_score = main_train(
        train_data_dir=data_dir,
        train_data_file = merged_txt,
        gen_type=module_type_upper,
        checkpoint_save_folder=checkpoint_save_folder,
        cuda_device_num=cuda_device_num,
        idx_iteration=idx_iteration,
        idx_response_id=idx_response_id
    )
    ############ comment those lines if you want to save training data and checkpoints ############
    try:
        if os.path.isdir(data_dir):
            if idx_iteration != 0 or idx_response_id != 0:
                shutil.rmtree(data_dir)
    except Exception as e:
        print(f"[WARN] Failed to remove data dir: {data_dir}, {e}")

    try:
        if os.path.isdir(checkpoint_save_folder):
            shutil.rmtree(checkpoint_save_folder)
    except Exception as e:
        print(f"[WARN] Failed to remove checkpoint folder: {checkpoint_save_folder}, {e}")
    ###############################################################################################
    
    if obj_type == 'min':
        return float(np.min(fitness_score))
    else:
        return float(np.max(fitness_score))

def module_type_lower(s):
    return s.lower()

# ==== main ====
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--idx_iteration", type=int, required=True)
    parser.add_argument("--idx_response_id", type=int, required=True)
    parser.add_argument("--cuda_device_num", type=int, required=True)
    parser.add_argument("--module_type", type=str, required=True)
    parser.add_argument("--n_proc", type=int, required=True)
    parser.add_argument("--obj_type", type=str, required=True)
    parser.add_argument("--n_total", type=int, required=True)
    parser.add_argument("--log_prefix", type=str, required=True)

    args = parser.parse_args()

    score = get_fitness_score(
        args.checkpoint_folder, args.idx_iteration, args.idx_response_id,
        args.cuda_device_num, args.module_type,
        n_total=args.n_total,
        n_proc=args.n_proc,
        log_prefix=args.log_prefix,
        obj_type=args.obj_type
    )
    print(json.dumps({"gap": score}), flush=True)



