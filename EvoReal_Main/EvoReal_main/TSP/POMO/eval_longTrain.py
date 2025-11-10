import os
import sys
import gc
import torch
import psutil
import json
##########################################################################################
# import
from POMO.TSPTester import TSPTester as Tester
from POMO.utils import *
##########################################################################################
# utilss
def check_memory():
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024 ** 3):.2f} GB")
    print(f"RAM Used: {mem.used / (1024 ** 3):.2f} GB")
    print(f"RAM Available: {mem.available / (1024 ** 3):.2f} GB")
    print(f"RAM Usage Rate: {mem.percent}%")
def log_memory():
    print("=" * 40)
    print(f"Allocated: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")
    print(f"Max Reserved: {torch.cuda.max_memory_reserved() / (1024**3):.2f} GB")
    print("=" * 40)
##############################
##########################################################################################
# Main Loop

def eval(val_root_dir,checkpoint_folder,idx_iteration,idx_response_id,cuda_device_num,module_type="R",epoch=0):

        # utils
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
        
        return None  
        
    DEBUG_MODE = False
    
    if torch.cuda.is_available():
        USE_CUDA = not DEBUG_MODE
    else:
        USE_CUDA = DEBUG_MODE

    CUDA_DEVICE_NUM = cuda_device_num
    # ##########################################################################################
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
        'test_episodes': 1,
        'test_batch_size': 1,
        'augmentation_enable': True,
        'aug_factor': 8,
        'aug_batch_size': 1,
        'finetune_mode':False,
        'epoch': epoch,
        
    }
    if tester_params['augmentation_enable']:
        tester_params['test_batch_size'] = tester_params['aug_batch_size']
    # Checkpoint setting
    tester_params['result_folder'] = checkpoint_folder
    tester_params['idx_iteration'] = idx_iteration
    tester_params['idx_response_id'] = idx_response_id
    

    json_files = sorted([f for f in os.listdir(val_root_dir) if f.endswith('.json')])
    tester_logger_dict = {}
    tester_params['num_samples'] = len(json_files)
    
    for problem_index, json_file in enumerate(json_files):
        file_path = os.path.join(val_root_dir, json_file)
        dimension = extract_dimension(file_path)
        file_name = os.path.splitext(json_file)[0]
        if dimension is None:
            print(f"Skipping {file_path}: No DIMENSION found.")
            continue


        env_params = {
            'problem_size': dimension,
            'pomo_size': dimension,
            'nonefinetune_path': file_path,
            'problem_idx': problem_index,
            'trainable': False,
            'module_type': module_type
        }

        print(f"Running test for {file_path} with problem size {dimension}")


        tester = Tester(env_params=env_params, model_params=model_params, tester_params=tester_params)
        score, aug_score = tester.run()

        if file_name not in tester_logger_dict:
            tester_logger_dict[file_name] = []
        tester_logger_dict[file_name].append([score, aug_score])

        # --- release momory ---
        del tester
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    label_dict = get_label()
    aug_gap, non_aug_gap = compute_gap(tester_logger_dict, label_dict)
    # result = {
    #     "aug_gap": round(aug_gap, 6),
    #     "non_aug_gap": round(non_aug_gap, 6)
    # }
    # print(json.dumps(result),file=sys.stdout)

    return aug_gap, non_aug_gap



