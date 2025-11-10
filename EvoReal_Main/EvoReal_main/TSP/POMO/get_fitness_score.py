import os
import sys
import gc
import torch
import psutil
import json
import argparse
import shutil
##########################################################################################
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# import train and eval module
# from POMO.eval_longTrain import eval as EVAL
from POMO.train_longTrain import train as TRAIN

# ##########################################################################################
tolerance = 3
early_stopping_flag = False
patience_counter = 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_folder", type=str, required=True)
    parser.add_argument("--idx_iteration", type=int, required=True)
    parser.add_argument("--idx_response_id", type=int, required=True)
    # parser.add_argument("--train_set_dir", type=str, required=True)
    parser.add_argument("--module_type", type=str, default="mixed")
    parser.add_argument("--cuda_device_num", type=int, required=True)
    parser.add_argument("--log_prefix", type=str, required=True)
    args = parser.parse_args()


    checkpoint_save_folder = os.path.join(args.checkpoint_folder,f"iter_{args.idx_iteration}_resp_{args.idx_response_id}")
    os.makedirs(checkpoint_save_folder, exist_ok=True)
    
    best_aug_gap, best_non_aug_gap = TRAIN(
        checkpoint_folder=checkpoint_save_folder,
        idx_iteration=args.idx_iteration,
        idx_response_id=args.idx_response_id,
        cuda_device_num=args.cuda_device_num,
        module_type=args.module_type,
        log_prefix = args.log_prefix,

    )
    try:
        shutil.rmtree(checkpoint_save_folder)
    except Exception as e:
        print(f"Warning: Failed to delete {checkpoint_save_folder}: {e}")
    
    return float(best_aug_gap), float(best_non_aug_gap)
if __name__ == "__main__":
    best_aug_gap, best_non_aug_gap = main()
    print(json.dumps({"aug_gap": best_aug_gap, "non_aug_gap": best_non_aug_gap}))