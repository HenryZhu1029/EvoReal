##########################################################################################
# Machine Environment Config
import torch
import os
import sys
import datetime
import time
DEBUG_MODE = False

USE_CUDA = torch.cuda.is_available() and not DEBUG_MODE

CUDA_DEVICE_NUM = 0
##########################################################################################
# Path Config

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import
import logging
from POMO.utils import get_num_samples

from POMO.TSPTrainer import TSPTrainer as Trainer
from POMO.TSPTester import TSPTester as Tester

##########################################################################################
# parameters
train_root_dir = './TSP_data/finetune_train_full'
val_root_dir = './TSP_data/same_train_as_evo'


num_train_samples = get_num_samples(train_root_dir)

num_val_samples = get_num_samples(val_root_dir)


num_epochs = 300 # total training epochs
################ Finetune Parameters(Global) ###################


global_params = {
    'total_epochs': num_epochs,
    
}
if DEBUG_MODE:
    total_epochs = 2
else:
    total_epochs = num_epochs
global_params['total_epochs'] = total_epochs

# Model Parameters
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

# Optimizer Parameters
optimizer_params = {
    'optimizer': {
        'lr': 1*1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [800],
        'gamma': 0.2
    }
}

#Logger Parameters
logger_params = {
    'log_file': {
        'desc': 'finetune_p2',
        'filename': 'run_log'
    }
}
##########################################################################################
#Trainer Parameters
train_env_params = {
'num_total_samples':num_train_samples,
'root_dir': train_root_dir,
'val_root_dir': val_root_dir,
'trainable': True
}
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': total_epochs,
    'use_early_stopping': True,
    'tolerance': 5, 
    'train_batch_size': 4,
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        'path': './POMO/result/p1-checkpoints',  # directory path of pre-trained model and log files saved.
        'epoch': 225, # epoch version of pre-trained model to laod.
        },
    'logging': {
        'model_save_interval': 10
        }

    }
if trainer_params['use_early_stopping']:
    trainer_params['stop_threshold'] = trainer_params['logging']['model_save_interval']/100*1
##########################################################################################
# main


def main():

    
    ## Pre-Training Logger Setup
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    logger.info('Optimizer Parameters:{}'.format(optimizer_params))
    logger.info('Trainer Parameters:{}'.format(trainer_params))

    start_time = time.time()


    trainer = Trainer(global_params=global_params,
                    env_params=train_env_params,
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    trainer_params=trainer_params)

    trainer.run()
    del trainer
    torch.cuda.empty_cache()




    logger.info(f"The finetuning procedure is done!")
    
    end_time = time.time()
    total_time_sec = end_time - start_time
    hours, rem = divmod(total_time_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    print(f"[TEST] Total execution time: {time_str}")




##########################################################################################
if __name__ == "__main__":
    main()
    log_filename = f"output_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    
    with open(log_filename, "w") as f:
        sys.stdout = f


