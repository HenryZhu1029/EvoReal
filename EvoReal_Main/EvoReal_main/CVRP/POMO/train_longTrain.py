##########################################################################################
# Import
import os
import sys
import gc
import torch
import psutil
import json
import logging
import time
from POMO.utils import create_logger
from POMO.CVRPTrainer import CVRPTrainer as Trainer

##########################################################################################
# utils
def get_num_samples(file_root_dir):
    "Get the number of TSP problems(in .pt form) in a folder"
    return len([f for f in os.listdir(file_root_dir) if f.endswith(".pt")])
##########################################################################################



def train(checkpoint_folder,idx_iteration,idx_response_id,cuda_device_num,module_type="cvrp",log_prefix=None):
    ##########################################################################################
    # Dataset Path Config
    val_root_dir = './dataset/val_set'
    
    # Device Setup
    DEBUG_MODE = False
        
    if torch.cuda.is_available():
        USE_CUDA = not DEBUG_MODE
    else:
        USE_CUDA = DEBUG_MODE
        
    CUDA_DEVICE_NUM = cuda_device_num
    
    # parameters
    num_training_samples = get_num_samples(val_root_dir)

    env_params = {
        'problem_size': 100,
        'pomo_size': 100,
        'num_total_samples': num_training_samples,
        'root_dir': val_root_dir,
        'use_early_stopping': False,
        'tolerance': 5,
        'log_prefix': log_prefix,
        'module_type': module_type,

    }

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

    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6
        },
        'scheduler': {
            'milestones': [100,110,120,130,140,150,160,170,180,190],
            'gamma': 0.9
        }
    }

    trainer_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'train_episodes': 500,
        'epochs': 40,
        'train_batch_size': 64,
        'prev_model_path': None,
        'start_early_stopping': 0,
        'use_optimizer_state': False,
        'logging': {
            'model_save_interval': 5,

        },
        'model_load': {
            'enable': True,  # enable loading pre-trained model
            'path': './POMO/result/pretrained_cvrp100_model',  # directory path of pre-trained model and log files saved.
            'epoch': 30500,  # epoch version of pre-trained model to laod.

        }
    }

    logger_params = {
        'log_file': {
            'desc': 'EvoReal_cvrp_pomo',
            'filename': 'proxy_eval_log'
        }
    }
    trainer_params['result_folder'] = checkpoint_folder
    trainer_params['idx_iteration'] = idx_iteration
    trainer_params['idx_response_id'] = idx_response_id
    
    
    if env_params['use_early_stopping']:
        env_params['early_stopping_threshold']=(trainer_params['logging']['model_save_interval']/100)*1
    ##########################################################################################
    create_logger(**logger_params)

    
    start_time = time.time()
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)


    best_aug_gap, best_non_aug_gap = trainer.run()
    
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[DEBUG] CUDA cache cleared.")
    return best_aug_gap, best_non_aug_gap
    # end_time = time.time()
    # total_time_sec = end_time - start_time
    # hours, rem = divmod(total_time_sec, 3600)
    # minutes, seconds = divmod(rem, 60)
    # time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    # print(f"[TEST] Total execution time: {time_str}")
    



# def _print_config():
#     logger = logging.getLogger('root')
#     logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
#     logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
#     [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

