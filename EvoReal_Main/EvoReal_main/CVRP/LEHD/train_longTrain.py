DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os

import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import logging
from LEHD.utils import create_logger
from LEHD.VRPTrainer import VRPTrainer as Trainer

##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')
mode = 'train'
training_data_path = b + '/data/vrp100_hgs_train_5w.txt'

env_params = {
    'data_path' : training_data_path,
    'mode': mode,
    'sub_path': True
}

model_params = {
    'mode': 'train',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
                 },
    'scheduler': {
        'milestones': [i for i in range(1,10)],
        'gamma': 0.9
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 10,
    'tolerance': 2,
    'train_episodes': 10000,
    'train_batch_size': 1024,
    'use_early_stopping': True,
    'use_optimizer_state': False,
    'logging': {
        'model_save_interval': 1,

               },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/20230817_235537_train',  # directory path of pre-trained model and log files saved.
        'epoch': 40,  # epoch version of pre-trained model to laod.
                  }
    }



##########################################################################################
# main

def main_train(train_data_dir = None, train_data_file = None, gen_type = 'CVRP', checkpoint_save_folder = None, cuda_device_num=None, idx_iteration=None, idx_response_id=None):
    
    logger_params = {
    'log_file': {
        'desc': 'cvrpdata_evo_lehd',
        'filename': f'Log{gen_type}_iter{idx_iteration}_resp{idx_response_id}.txt'
    }
    }
    if DEBUG_MODE:
        _set_debug_mode()
    if cuda_device_num is not None:
        trainer_params['cuda_device_num'] = cuda_device_num
    if checkpoint_save_folder is not None:
        trainer_params['checkpoint_save_folder'] = checkpoint_save_folder
        
    if gen_type is not None:
        trainer_params['gen_type'] = gen_type
    if train_data_dir is not None and gen_type is not None:
        env_params['train_data_dir'] = train_data_dir
        env_params['data_path'] = train_data_file
        
    env_params['idx_iteration'] = idx_iteration
    env_params['idx_response_id'] = idx_response_id
    
    create_logger(**logger_params)
    
    if idx_iteration ==0:
         _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)


    saved_gap = trainer.run()

    return saved_gap


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 4
    trainer_params['train_episodes'] = 100
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main_train()
