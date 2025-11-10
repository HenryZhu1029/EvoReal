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
from LEHD_CVRP_p2.utils.utils import create_logger
from LEHD_CVRP_p2.CVRP.VRPTrainer import VRPTrainer as Trainer

##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')

training_data_path = b + '/data/SetX_train70_LEHD.txt' # realistic training set

env_params = {
    'data_path' : training_data_path,
    'mode': 'train',
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
        'lr': 1e-5,
                 },
    'scheduler': {
        'milestones': [i for i in range(1,100)],
        'gamma': 0.8
                 }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 50,
    'tolerance': 5,
    'train_episodes': 70,
    'max_batch_size': 16,
    'logging': {
        'model_save_interval': 1,

               },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './result/p1-checkpoints',  # directory path of pre-trained model from phase one.
        'epoch': 40,  # epoch version of pre-trained model to laod.
                  }
    }

logger_params = {
    'log_file': {
        'desc': 'p2_finetune_lehd_cvrp',
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

    # copy_all_src(trainer.result_folder)

    trainer.run()


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
    main()

