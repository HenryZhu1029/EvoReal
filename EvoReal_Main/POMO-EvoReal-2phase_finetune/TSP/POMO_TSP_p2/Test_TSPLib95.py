"""
This .py file is not used for standalone inference. It is part of the validation module in the finetuning process.

"""


import os
import sys
import logging
import torch
import psutil
import json
import psutil
##########################################################################################
# Path Config

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

from POMO.utils import create_logger,get_num_samples
from POMO.TSPTester import TSPTester as Tester
from POMO.Problem_Def import extract_dimension
##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
if torch.cuda.is_available():
    USE_CUDA = not DEBUG_MODE
else:
    USE_CUDA = DEBUG_MODE

CUDA_DEVICE_NUM = 0



##########################################################################################
# Logger Setup

logger_params = {
    'log_file': {
        'desc': 'test_on_TSPLib95',
        'filename': 'run_log'
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
        'path': './POMO/result/saved_tsp100_model',
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
json_save_path = './test_logger_dict_tsplib70'

if __name__ == "__main__":
    tester_logger_dict = main_test()
    with open(json_save_path,'w') as f:
        json.dump(tester_logger_dict,f,indent=2)


