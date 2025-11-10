##########################################################################################
# import
import torch
import gc
# import module
from POMO.TSPTrainer import TSPTrainer as Trainer

##########################################################################################
# main

def train(checkpoint_folder,idx_iteration,idx_response_id,cuda_device_num,module_type="S1",log_prefix=None):
    ##########################################################################################
# Machine Environment Config
    
    DEBUG_MODE = False
    train_root_dir = f'./dataset/{module_type.upper()}_spectral'
    
    if torch.cuda.is_available():
        USE_CUDA = not DEBUG_MODE
    else:
        USE_CUDA = DEBUG_MODE
        
    CUDA_DEVICE_NUM = cuda_device_num
    
    # parameters
    
    env_params = {
        'problem_size': 100,
        'pomo_size': 100,
        'trainable': True,
        "module_type": module_type,
        'train_root_dir': train_root_dir,
        'log_prefix': log_prefix
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
            'lr': 1*1e-4,
            'weight_decay': 1e-6
        },
        'scheduler': {
            'milestones': [50],
            'gamma': 0.2
        }
    }

    trainer_params = {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'epochs': 40,
        'train_episodes': 5000,
        'train_batch_size': 64,
        'tolerance': 5,
        'use_early_stopping': True,
        'use_optimizer_state': False,
        'model_load': {
            'enable': True,  # enable loading pre-trained model
            'path': './POMO/result/saved_tsp100_model',  # directory path of pre-trained model and log files saved.
            'epoch': 2000,  # epoch version of pre-trained model to laod.
            },
        'model_save_interval': 1
        
    }
    
    if DEBUG_MODE:
        trainer_params['epochs'] = 2
        trainer_params['train_episodes'] = 10
        trainer_params['train_batch_size'] = 4
    
    if env_params['module_type'] == 'S1':
        trainer_params['epochs'] = 40
        trainer_params['start_early_stopping'] = 30
    elif env_params['module_type'] == 'S2':
        trainer_params['epochs'] = 40
        trainer_params['start_early_stopping'] = 20
    elif env_params['module_type'] == 'S3':
        trainer_params['epochs'] = 40
        trainer_params['start_early_stopping'] = 10
    else:
        trainer_params['epochs'] = 5
        trainer_params['start_early_stopping'] = 0
    # _print_config()
    trainer_params['result_folder'] = checkpoint_folder
    trainer_params['idx_iteration'] = idx_iteration
    trainer_params['idx_response_id'] = idx_response_id
    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    best_aug_gap, best_non_aug_gap = trainer.run()



    # --- Manual cleanup to avoid memory leaks ---
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("[DEBUG] CUDA cache cleared.")
    return best_aug_gap, best_non_aug_gap

# def _print_config():
#     logger = logging.getLogger('root')
#     logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
#     logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
#     [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


