
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
from LEHD_TSP_p1.utils.utils import create_logger
from LEHD_TSP_p1.TSP.TSPTrainer import TSPTrainer as Trainer

##########################################################################################
# parameters

b = os.path.abspath(".").replace('\\', '/')

gen_type = 'mixed' #must in upper case
mode = 'train'
training_data_path = b+f"/data/tsp100_type{gen_type}_merged.txt"

env_params = {
    'data_path':training_data_path,
    'mode': mode,
    'sub_path': True,

}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':6,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-5,
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 20)],
        'gamma': 0.6
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 20,
    'tolerance': 3,
    'use_optimizer_state':False,
    'use_early_stopping': True,
    'gen_type': gen_type,
    'train_episodes': 50000,
    'train_batch_size': 1024,
    'logging': {
        'model_save_interval': 1,
               },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/20230509_153705_train',  # directory path of pre-trained model and log files saved.
        'epoch': 150,  # epoch version of pre-trained model to laod.
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'finetune_p1',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)


    trainer.run()


def _set_debug_mode():
    global trainer_params

    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 8
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
