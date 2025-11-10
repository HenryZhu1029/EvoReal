import os
import sys
import logging
import torch
import psutil
import json
import psutil
import numpy as np
##########################################################################################
# Path Config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "POMO_TSP_p2"))

from POMO_TSP_p2.POMO.utils import create_logger,get_num_samples
from POMO_TSP_p2.POMO.TSPTester import TSPTester as Tester
from POMO_TSP_p2.POMO.Problem_Def import extract_dimension
##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
if torch.cuda.is_available():
    USE_CUDA = not DEBUG_MODE
else:
    USE_CUDA = DEBUG_MODE

CUDA_DEVICE_NUM = 0

def get_label(label_path="./tsplib_labels.json"):
    with open(label_path, "r") as f:
        return json.load(f)

def compute_gap(tester_logger_dict, label_dict):
    per_problem_results = {}  

    gap_in_0_200_aug = []
    gap_in_0_200_non_aug = []
    gap_in_200_500_aug = []
    gap_in_200_500_non_aug = []
    gap_in_500_1000_aug = []
    gap_in_500_1000_non_aug = []
    gap_in_1000_5000_aug = []
    gap_in_1000_5000_non_aug = []

    total_aug_gap = 0
    total_non_aug_gap = 0
    count = 0

    for problem_name in tester_logger_dict:
        if problem_name not in label_dict:
            continue
        opt_norm = label_dict[problem_name]["opt"]
        opt_real = label_dict[problem_name].get("Opt", None)
        if opt_real is None or opt_norm == 0:
            continue 
        problem_size = label_dict[problem_name]["Problem Size"]
        non_aug_score_norm, aug_score_norm = tester_logger_dict[problem_name][0]

        aug_score_real = (opt_real / opt_norm) * aug_score_norm
        non_aug_score_real = (opt_real / opt_norm) * non_aug_score_norm

        aug_gap = (aug_score_norm - opt_norm) / opt_norm
        non_aug_gap = (non_aug_score_norm - opt_norm) / opt_norm

        per_problem_results[problem_name] = {
            "problem_size": problem_size,
            "Opt": opt_real,   # optimum from official tsplib dataset
            "aug_score_real": aug_score_real,
            "non_aug_score_real": non_aug_score_real,
            "aug_gap": aug_gap,
            "non_aug_gap": non_aug_gap,
        }

        total_aug_gap += aug_gap
        total_non_aug_gap += non_aug_gap

        if problem_size > 0 and problem_size < 200:
            gap_in_0_200_aug.append(aug_gap)
            gap_in_0_200_non_aug.append(non_aug_gap)
        elif problem_size >= 200 and problem_size < 500:
            gap_in_200_500_aug.append(aug_gap)
            gap_in_200_500_non_aug.append(non_aug_gap)
        elif problem_size >= 500 and problem_size < 1000:
            gap_in_500_1000_aug.append(aug_gap)
            gap_in_500_1000_non_aug.append(non_aug_gap)
        else:
            gap_in_1000_5000_aug.append(aug_gap)
            gap_in_1000_5000_non_aug.append(non_aug_gap)

        count += 1

    if count == 0:
        return None, None, None

    # means
    avg_result = {
        "avg_aug_gap_all": total_aug_gap / count,
        "avg_non_aug_gap_all": total_non_aug_gap / count,
        "avg_aug_gap_0_200": np.mean(gap_in_0_200_aug) if gap_in_0_200_aug else None,
        "avg_non_aug_gap_0_200": np.mean(gap_in_0_200_non_aug) if gap_in_0_200_non_aug else None,
        "avg_aug_gap_200_500": np.mean(gap_in_200_500_aug) if gap_in_200_500_aug else None,
        "avg_non_aug_gap_200_500": np.mean(gap_in_200_500_non_aug) if gap_in_200_500_non_aug else None,
        "avg_aug_gap_500_1000": np.mean(gap_in_500_1000_aug) if gap_in_500_1000_aug else None,
        "avg_non_aug_gap_500_1000": np.mean(gap_in_500_1000_non_aug) if gap_in_500_1000_non_aug else None,
        "avg_aug_gap_1000_5000": np.mean(gap_in_1000_5000_aug) if gap_in_1000_5000_aug else None,
        "avg_non_aug_gap_1000_5000": np.mean(gap_in_1000_5000_non_aug) if gap_in_1000_5000_non_aug else None,
    }

    return per_problem_results, avg_result, count

##########################################################################################
# Logger Setup

logger_params = {
    'log_file': {
        'desc': 'test_on_TSPLib95',
        'filename': 'test_result'
    }
}
create_logger(**logger_params)
logger = logging.getLogger('root')

##########################################################################################
# Constants


model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': None,
        'epoch': None,
    },
    'test_episodes': 1,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'finetune_mode':False,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

##########################################################################################

# Main Loop

def main_test(problem_dir=None,checkpoint_path=None,epoch=None):
    
    root_dir = problem_dir
    num_test_samples = get_num_samples(problem_dir)   
    if checkpoint_path is not None and epoch is not None:
        tester_params['model_load']['path'] = checkpoint_path
        tester_params['model_load']['epoch'] = epoch
        
    if isinstance(num_test_samples,int) and num_test_samples>0:
        tester_params['num_samples'] = num_test_samples
        
    txt_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.txt')])
    tester_logger_dict = {}
    for problem_index, txt_file in enumerate(txt_files):
        file_path = os.path.join(root_dir, txt_file)
        dimension = extract_dimension(file_path)
        file_name = os.path.splitext(txt_file)[0]
        if dimension is None:
            logger.warning(f"Skipping {file_path}: No DIMENSION found.")
            continue
    

        env_params = {
            'problem_size': dimension,
            'pomo_size': dimension,
            'problem_path': file_path,
            'root_dir': root_dir,
            'epoch_idx': epoch,
            'problem_idx': problem_index,
            'trainable':False
        }

        logger.info(f"Running test for {file_path} with problem size {dimension}")

        tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)

        score,aug_score = tester.run()
        if file_name not in tester_logger_dict:
            tester_logger_dict[file_name] = []
            tester_logger_dict[file_name].append([score, aug_score])
        del tester  # delete tester
        torch.cuda.empty_cache()  # clean ram fractions
    return tester_logger_dict

import json
json_save_path = './test_tsplib_statistics_pomo.json'
problem_dir = './POMO_TSP_p2/TSP_Sorted_new/full_problems'
checkpoint_path = "./POMO_TSP_p2/POMO/result/saved_p2_checkpoints"   # progressively finetuned model 
epoch = 290
if __name__ == "__main__":
    tester_logger_dict = main_test(problem_dir=problem_dir,checkpoint_path=checkpoint_path,epoch=epoch)
    label_dict = get_label()

    per_problem_gaps, avg_result, count = compute_gap(tester_logger_dict, label_dict)
    with open(json_save_path, 'w') as f:
        json.dump({
            "per_problem_gaps": per_problem_gaps,
            "avg_gap_result": avg_result,
            "count": count
        }, f, indent=2)
    print(f'The complete test statistics has been saved to {json_save_path}!')



